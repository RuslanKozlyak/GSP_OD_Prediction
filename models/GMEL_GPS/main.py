import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
from tqdm.auto import tqdm

from .model import GMEL_GPS
from .config import (
    GmelGpsConfig, WEIGHTS_DIR, device, ensure_dirs,
    save_model_weights, save_metrics_to_csv,
)
from models.GPS.metrics import compute_metrics


# ─── Inference helper ────────────────────────────────────────────────────────

def predict_gmel_gps(model, gbrt, city_data, dev=None):
    """Run GPS encoders then GBRT for full N×N OD prediction.

    Args:
        model:     trained GMEL_GPS instance
        gbrt:      trained GradientBoostingRegressor
        city_data: dict from prepare_single_city_data (needs 'graph_data',
                   'distances_scaled', 'num_nodes')
        dev:       torch.device (defaults to model's device)

    Returns:
        pred: (N, N) numpy array, non-negative
    """
    if dev is None:
        dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        _, _, _, h_in, h_out = model(city_data['graph_data'])
    n = h_in.shape[0]
    h_in_np  = h_in.cpu().numpy()   # (N, hd)
    h_out_np = h_out.cpu().numpy()  # (N, hd)
    # Build all N² feature vectors: [h_in[i] ‖ h_out[j] ‖ dis[i,j]]
    h_o  = h_in_np.reshape(n, 1, -1).repeat(n, axis=1)   # (N, N, hd)
    h_d  = h_out_np.reshape(1, n, -1).repeat(n, axis=0)  # (N, N, hd)
    dis  = city_data['distances_scaled'].reshape(n, n, 1) # (N, N, 1)
    feat = np.concatenate([h_o, h_d, dis], axis=2).reshape(-1, h_in_np.shape[1] * 2 + 1)
    pred = gbrt.predict(feat).reshape(n, n)
    pred[pred < 0] = 0
    return pred


# ─── Training ────────────────────────────────────────────────────────────────

def train(run_id, run_name, config, city_data):
    """Train GMEL_GPS (GPS encoders + GBRT decoder) on a single city.

    Phase 1 — GPS encoders trained with GMEL-style multitask MSE loss:
        L = MSE(flow_in.squeeze(1),  od_train.sum(0))
          + MSE(flow_out.squeeze(1), od_train.sum(1))
          + MSE(flow,                od_train)       ← (N,1) vs (N,N) broadcast

    Phase 2 — GBRT fitted on frozen GPS embeddings:
        features[i,j] = [h_in[i] ‖ h_out[j] ‖ distance[i,j]]   (N², 2·hd+1)
        targets       = od_train.ravel()

    Saves:
        results/weights/{run_id}.pt          — GMEL_GPS state_dict
        results/weights/{run_id}.json        — GmelGpsConfig serialised
        results/weights/{run_id}_gbrt.joblib — GBRT model
        results/metrics.csv                  — appended row

    Args:
        run_id:    unique identifier string
        run_name:  human-readable name
        config:    GmelGpsConfig instance
        city_data: dict from models.GPS.data_load.prepare_single_city_data

    Returns:
        dict with keys: name, model, gbrt, config, history,
                        metrics_full, metrics_nonzero, metrics_test_pairs, status
    """
    ensure_dirs()

    gd        = city_data['graph_data']
    od_train  = city_data['od_matrix_train'].astype(float)   # (N, N)
    od_np     = city_data['od_matrix_np'].astype(float)      # (N, N)
    val_mask  = city_data['val_mask']                        # (N, N) bool
    test_mask = city_data['test_mask']                       # (N, N) bool
    dis       = city_data['distances_scaled']                # (N, N)

    od_t     = torch.FloatTensor(od_train).to(device)
    od_val_t = torch.FloatTensor(od_np * val_mask).to(device)

    # ── Build model ──────────────────────────────────────────────────────────
    model = GMEL_GPS(
        input_dim  = gd.x.shape[1],
        edge_dim   = gd.edge_attr.shape[1],
        hidden_dim = config.hidden_dim,
        pe_dim     = config.pe_dim,
        n_layers   = config.n_layers,
        n_heads    = config.n_heads,
        dropout    = config.dropout,
        pe_type    = config.pe_type,
        norm_type  = config.gps_norm_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}\n  {run_name}\n  {config.describe()}\n{'='*70}")
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss  = float('inf')
    patience_count = 0
    best_state     = None
    status         = 'ok'
    epoch          = 0

    # ── Phase 1: train GPS encoders ──────────────────────────────────────────
    pbar = tqdm(range(1, config.max_epochs + 1), desc='GMEL_GPS', unit='ep')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        flow_in, flow_out, flow, _, _ = model(gd)
        loss = (F.mse_loss(flow_in.squeeze(1),  od_t.sum(0))
              + F.mse_loss(flow_out.squeeze(1), od_t.sum(1))
              + F.mse_loss(flow,                od_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vfi, vfo, vf, _, _ = model(gd)
            val_loss = (F.mse_loss(vfi.squeeze(1),  od_val_t.sum(0))
                      + F.mse_loss(vfo.squeeze(1), od_val_t.sum(1))
                      + F.mse_loss(vf,              od_val_t)).item()

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            flag = ' *'
        else:
            patience_count += 1
            flag = ''

        if epoch % 10 == 0 or epoch == 1:
            pbar.write(f"  {epoch:3d}/{config.max_epochs}  "
                       f"train={loss.item():.4g}  val={val_loss:.4g}  "
                       f"pat={patience_count}{flag}")

        if patience_count >= config.patience:
            print(f"  Early stop @ epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # ── Phase 2: GBRT on frozen GPS embeddings ────────────────────────────────
    print('  GMEL_GPS: extracting embeddings...')
    model.eval()
    with torch.no_grad():
        _, _, _, h_in, h_out = model(gd)
        h_in_np  = h_in.cpu().numpy()
        h_out_np = h_out.cpu().numpy()

    n = h_in_np.shape[0]
    h_o  = h_in_np.reshape(n, 1, -1).repeat(n, axis=1)
    h_d  = h_out_np.reshape(1, n, -1).repeat(n, axis=0)
    feat = np.concatenate([h_o, h_d, dis.reshape(n, n, 1)], axis=2)
    feat = feat.reshape(-1, h_in_np.shape[1] * 2 + 1)

    print(f'  GMEL_GPS: fitting GBRT on {feat.shape[0]:,} pairs ...')
    gbrt = GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        min_samples_split=2,
        min_samples_leaf=2,
        max_depth=None,
    )
    gbrt.fit(feat, od_train.reshape(-1))
    print('  GMEL_GPS: GBRT fitted.')

    # Save GBRT
    gbrt_path = WEIGHTS_DIR / f"{run_id}_gbrt.joblib"
    joblib.dump(gbrt, str(gbrt_path))
    print(f"  -> GBRT saved to {gbrt_path}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    pred  = predict_gmel_gps(model, gbrt, city_data, device)
    nz    = od_np > 0

    mf  = compute_metrics(pred.ravel(),        od_np.ravel())
    mnz = compute_metrics(pred[nz],            od_np[nz])
    mt  = compute_metrics(pred[test_mask],     od_np[test_mask])

    print(f"\n  === Evaluation ===")
    print(f"    CPC_full={mf['CPC']:.4f}  CPC_nz={mnz['CPC']:.4f}  "
          f"CPC_test={mt['CPC']:.4f}  MAE={mf['MAE']:.4f}")

    # ── Save weights & metrics ────────────────────────────────────────────────
    save_metrics_to_csv(run_id, run_name, config, mf, mnz, mt,
                        n_params, epoch, status)
    save_model_weights(run_id, model, config)

    return {
        'name':               run_name,
        'model':              model,
        'gbrt':               gbrt,
        'config':             config,
        'history':            history,
        'metrics_full':       mf,
        'metrics_nonzero':    mnz,
        'metrics_test_pairs': mt,
        'status':             status,
    }
