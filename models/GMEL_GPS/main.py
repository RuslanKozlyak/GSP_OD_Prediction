import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
from tqdm.auto import tqdm

from .model import GMEL_GPS
from models.GPS.config import (
    TrainingConfig, WEIGHTS_DIR, device, ensure_dirs,
    save_model_weights, save_metrics_to_csv,
)
from models.shared.metrics import compute_metrics


# ─── Inference helper ────────────────────────────────────────────────────────

def predict_gmel_gps(model, decoder, city_data, dev=None):
    """Run GPS encoders then decoder (GBRT or LGBM) for full N×N OD prediction."""
    if dev is None:
        dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        _, _, _, h_in, h_out = model(city_data['graph_data'])
    n = h_in.shape[0]
    h_in_np = h_in.cpu().numpy()
    h_out_np = h_out.cpu().numpy()
    h_o = h_out_np.reshape(n, 1, -1).repeat(n, axis=1)
    h_d = h_in_np.reshape(1, n, -1).repeat(n, axis=0)
    dis = city_data['distances_scaled'].reshape(n, n, 1)
    feat = np.concatenate([h_o, h_d, dis], axis=2).reshape(-1, h_in_np.shape[1] * 2 + 1)
    pred = decoder.predict(feat).reshape(n, n)
    pred[pred < 0] = 0
    return pred


# ─── Training ────────────────────────────────────────────────────────────────

def train(run_id, run_name, config, city_data):
    """Train GMEL_GPS (GPS encoders + GBRT/LGBM decoder) on a single city.

    Args:
        run_id:    unique identifier string
        run_name:  human-readable name
        config:    TrainingConfig instance (decoder_type should be 'gbrt' or 'lgbm')
        city_data: dict from models.GPS.data_load.prepare_single_city_data
    """
    ensure_dirs()

    gd = city_data['graph_data']
    od_train = city_data['od_matrix_train'].astype(float)
    od_np = city_data['od_matrix_np'].astype(float)
    val_mask = city_data['val_mask']
    test_mask = city_data['test_mask']
    dis = city_data['distances_scaled']

    od_t = torch.FloatTensor(od_train).to(device)
    od_val_t = torch.FloatTensor(od_np * val_mask).to(device)

    # ── Build model ──────────────────────────────────────────────────────────
    hidden_dim = 64
    pe_dim = 8
    n_layers = 3
    n_heads = 4
    dropout = 0.1

    model = GMEL_GPS(
        input_dim=gd.x.shape[1],
        edge_dim=gd.edge_attr.shape[1],
        hidden_dim=hidden_dim,
        pe_dim=pe_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        pe_type=config.pe_type,
        norm_type=config.gps_norm_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}\n  {run_name}\n  {config.describe()}\n{'='*70}")
    print(f"  Params: {n_params:,}")

    max_epochs = 300
    patience_limit = 20
    lr = config.learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_count = 0
    best_state = None
    status = 'ok'
    epoch = 0

    # ── Phase 1: train GPS encoders ──────────────────────────────────────────
    pbar = tqdm(range(1, max_epochs + 1), desc='GMEL_GPS', unit='ep')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        flow_in, flow_out, flow, _, _ = model(gd)
        loss = (F.mse_loss(flow_in.squeeze(1), od_t.sum(0))
                + F.mse_loss(flow_out.squeeze(1), od_t.sum(1))
                + F.mse_loss(flow, od_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vfi, vfo, vf, _, _ = model(gd)
            val_loss = (F.mse_loss(vfi.squeeze(1), od_val_t.sum(0))
                        + F.mse_loss(vfo.squeeze(1), od_val_t.sum(1))
                        + F.mse_loss(vf, od_val_t)).item()

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            flag = ' *'
        else:
            patience_count += 1
            flag = ''

        if epoch % 10 == 0 or epoch == 1:
            pbar.write(f"  {epoch:3d}/{max_epochs}  "
                       f"train={loss.item():.4g}  val={val_loss:.4g}  "
                       f"pat={patience_count}{flag}")

        if patience_count >= patience_limit:
            print(f"  Early stop @ epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # ── Phase 2: fit decoder on frozen GPS embeddings ────────────────────────
    print('  GMEL_GPS: extracting embeddings...')
    model.eval()
    with torch.no_grad():
        _, _, _, h_in, h_out = model(gd)
        h_in_np = h_in.cpu().numpy()
        h_out_np = h_out.cpu().numpy()

    n = h_in_np.shape[0]
    h_o = h_out_np.reshape(n, 1, -1).repeat(n, axis=1)
    h_d = h_in_np.reshape(1, n, -1).repeat(n, axis=0)
    feat = np.concatenate([h_o, h_d, dis.reshape(n, n, 1)], axis=2)
    feat = feat.reshape(-1, h_in_np.shape[1] * 2 + 1)

    y_all = od_train.reshape(-1)
    if config.include_zero_pairs:
        X_fit, y_fit = feat, y_all
        print(f'  GMEL_GPS: fitting {config.decoder_type.upper()} on '
              f'{X_fit.shape[0]:,} pairs (all) ...')
    else:
        nz_mask = y_all > 0
        X_fit, y_fit = feat[nz_mask], y_all[nz_mask]
        print(f'  GMEL_GPS: fitting {config.decoder_type.upper()} on '
              f'{X_fit.shape[0]:,} nonzero pairs ...')

    if config.decoder_type == 'lgbm':
        import lightgbm as lgb
        val_flat = (od_np * val_mask).reshape(-1)
        val_nz = val_flat > 0
        X_val_fit = feat[val_nz]
        y_val_fit = val_flat[val_nz]
        lgbm_params = {
            'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.05,
            'num_leaves': config.lgbm_num_leaves, 'max_depth': 8,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'verbose': -1, 'seed': 42,
        }
        decoder = lgb.train(
            lgbm_params,
            lgb.Dataset(X_fit, y_fit),
            num_boost_round=config.lgbm_n_estimators,
            valid_sets=[lgb.Dataset(X_val_fit, y_val_fit)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )
        decoder_path = WEIGHTS_DIR / f"{run_id}_lgbm.lgbm"
        decoder.save_model(str(decoder_path))
        print(f'  GMEL_GPS: LGBM fitted. -> {decoder_path}')

    else:  # 'gbrt'
        decoder = GradientBoostingRegressor(
            n_estimators=config.gbrt_n_estimators,
            min_samples_split=2,
            min_samples_leaf=2,
            max_depth=None,
        )
        decoder.fit(X_fit, y_fit)
        decoder_path = WEIGHTS_DIR / f"{run_id}_gbrt.joblib"
        joblib.dump(decoder, str(decoder_path))
        print(f'  GMEL_GPS: GBRT fitted. -> {decoder_path}')

    # ── Evaluation ───────────────────────────────────────────────────────────
    pred = predict_gmel_gps(model, decoder, city_data, device)
    nz = od_np > 0

    mf = compute_metrics(pred.ravel(), od_np.ravel())
    mnz = compute_metrics(pred[nz], od_np[nz])
    mt = compute_metrics(pred[test_mask], od_np[test_mask])

    print(f"\n  === Evaluation ===")
    print(f"    CPC_full={mf['CPC']:.4f}  CPC_nz={mnz['CPC']:.4f}  "
          f"CPC_test={mt['CPC']:.4f}  MAE={mf['MAE']:.4f}")

    save_metrics_to_csv(run_id, run_name, config, mf, mnz, mt,
                        n_params, epoch, status)
    save_model_weights(run_id, model, config)

    return {
        'name': run_name,
        'model': model,
        'decoder': decoder,
        'config': config,
        'history': history,
        'metrics_full': mf,
        'metrics_nonzero': mnz,
        'metrics_test_pairs': mt,
        'status': status,
    }
