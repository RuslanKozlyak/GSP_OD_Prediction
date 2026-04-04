import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from .model import GMEL_GPS
from models.GPS.config import (
    TrainingConfig, WEIGHTS_DIR, device, ensure_dirs,
    save_model_weights, save_metrics_to_csv,
)
from models.shared.metrics import cal_od_metrics, compute_metrics
from models.shared.plotting import save_loss_plot


def _masked_mse(pred, target, mask):
    """Compute MSE only on observed entries to avoid supervising held-out pairs as zero."""
    if mask is None:
        return F.mse_loss(pred, target)
    if mask.sum().item() == 0:
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[mask], target[mask])


def _marginal_mse(pred, target):
    # Keep marginals as 1D vectors; otherwise (N, 1) vs (N,) broadcasts to (N, N).
    if pred.dim() == target.dim() + 1 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if pred.shape != target.shape:
        raise ValueError(f"Marginal shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    return F.mse_loss(pred, target)


def _scale_masked_matrix(od_matrix, fit_mask, apply_mask):
    """Fit scaler on train pairs, then fill only the requested mask with scaled values."""
    fit_values = od_matrix[fit_mask].reshape(-1, 1)
    if fit_values.size == 0:
        fit_values = od_matrix.reshape(-1, 1)
    fit_values = np.concatenate([fit_values, np.zeros((1, 1), dtype=fit_values.dtype)], axis=0)
    scaler = MinMaxScaler().fit(fit_values)
    scaled = np.zeros_like(od_matrix, dtype=np.float32)
    if apply_mask.any():
        scaled[apply_mask] = scaler.transform(
            od_matrix[apply_mask].reshape(-1, 1)
        ).reshape(-1)
    return scaled, scaler


def _build_decoder_training_set(feat, od_matrix, train_mask, include_zero_pairs, zero_pair_ratio):
    """Fit the tree decoder on train positives, optionally mixing in true zeros only."""
    train_idx = np.flatnonzero(train_mask.reshape(-1))
    y_flat = od_matrix.reshape(-1)

    if train_idx.size == 0:
        return feat, y_flat

    if not include_zero_pairs:
        return feat[train_idx], y_flat[train_idx]

    zero_idx = np.flatnonzero(y_flat == 0)
    if zero_idx.size == 0:
        return feat[train_idx], y_flat[train_idx]

    zero_ratio = float(np.clip(zero_pair_ratio, 0.0, 0.95))
    n_zero = int(round(train_idx.size * zero_ratio / max(1e-8, 1.0 - zero_ratio)))
    n_zero = min(n_zero, zero_idx.size)
    if n_zero <= 0:
        return feat[train_idx], y_flat[train_idx]

    rng = np.random.default_rng(42)
    sampled_zero_idx = rng.choice(zero_idx, size=n_zero, replace=False)
    fit_idx = np.concatenate([train_idx, sampled_zero_idx])
    return feat[fit_idx], y_flat[fit_idx]


def _predict_bilinear_matrix(model, city_data, od_scaler):
    model.eval()
    with torch.no_grad():
        _, _, flow, _, _ = model(city_data['graph_data'])
    pred = od_scaler.inverse_transform(flow.detach().cpu().numpy().reshape(-1, 1)).reshape(flow.shape)
    pred[pred < 0] = 0
    return pred


def _print_stage_metrics(stage_name, pred, od_np, test_mask):
    nz = od_np > 0
    mf = cal_od_metrics(pred, od_np)
    mnz = compute_metrics(pred[nz], od_np[nz]) if np.any(nz) else {'CPC': 0.0, 'MAE': 0.0, 'RMSE': 0.0}
    mt = compute_metrics(pred[test_mask], od_np[test_mask]) if np.any(test_mask) else {'CPC': 0.0, 'MAE': 0.0, 'RMSE': 0.0}
    print(f"\n  === {stage_name} ===")
    print(f"    CPC_full={mf['CPC']:.4f}  CPC_nz={mnz['CPC']:.4f}  "
          f"CPC_test={mt['CPC']:.4f}  MAE={mf['MAE']:.4f}  RMSE={mf['RMSE']:.4f}")
    print(f"    Full metrics: {mf}")
    print(f"    Nonzero metrics: {mnz}")
    print(f"    Test-pair metrics: {mt}")
    return mf, mnz, mt


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
    train_mask = city_data['train_mask']
    val_mask = city_data['val_mask']
    test_mask = city_data['test_mask']
    dis = city_data['distances_scaled']

    od_train_scaled, od_scaler = _scale_masked_matrix(od_np, train_mask, train_mask)
    od_val_scaled, _ = _scale_masked_matrix(od_np, train_mask, val_mask)
    od_t = torch.FloatTensor(od_train_scaled).to(device)
    od_val_t = torch.FloatTensor(od_val_scaled).to(device)
    train_mask_t = torch.BoolTensor(train_mask).to(device)
    val_mask_t = torch.BoolTensor(val_mask).to(device)

    # Full-matrix marginals — must come from the complete OD, not the masked version
    od_full_scaled = od_scaler.transform(od_np.reshape(-1, 1)).reshape(od_np.shape)
    marginal_in_t = torch.FloatTensor(od_full_scaled.sum(0)).to(device)
    marginal_out_t = torch.FloatTensor(od_full_scaled.sum(1)).to(device)

    # ── Build model ──────────────────────────────────────────────────────────
    model = GMEL_GPS(
        input_dim=gd.x.shape[1],
        edge_dim=gd.edge_attr.shape[1],
        pe_type=config.pe_type,
        norm_type=config.gps_norm_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}\n  {run_name}\n  {config.describe()}\n{'='*70}")
    print(f"  Params: {n_params:,}")
    if config.loss_type != 'multitask' or config.prediction_mode != 'raw':
        print("  Note: GMEL_GPS encoder pretraining uses masked MSE on OD+marginals;"
              " loss_type/prediction_mode are metadata only here.")

    max_epochs = config.epochs
    patience_limit = config.patience
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
    loss_plot_path = WEIGHTS_DIR.parent / "loss_plots" / f"{run_id}_loss.png"

    # ── Phase 1: train GPS encoders ──────────────────────────────────────────
    pbar = tqdm(range(1, max_epochs + 1), desc='GMEL_GPS', unit='ep')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        flow_in, flow_out, flow, _, _ = model(gd)
        loss = (_marginal_mse(flow_in, marginal_in_t)
                + _marginal_mse(flow_out, marginal_out_t)
                + _masked_mse(flow, od_t, train_mask_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vfi, vfo, vf, _, _ = model(gd)
            val_loss = (_marginal_mse(vfi, marginal_in_t)
                        + _marginal_mse(vfo, marginal_out_t)
                        + _masked_mse(vf, od_val_t, val_mask_t)).item()

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

    saved_plot_path = save_loss_plot(
        history['train_loss'],
        history['val_loss'],
        title=f"{run_name} Loss",
        save_path=loss_plot_path,
    )
    if saved_plot_path is not None:
        print(f"  -> Loss plot saved to {saved_plot_path}")
        model.loss_plot_path = str(saved_plot_path)

    bilinear_pred = _predict_bilinear_matrix(model, city_data, od_scaler)
    bilinear_mf, bilinear_mnz, bilinear_mt = _print_stage_metrics(
        "Bilinear Head", bilinear_pred, od_np, test_mask
    )

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

    X_fit, y_fit = _build_decoder_training_set(
        feat,
        od_np,
        train_mask,
        include_zero_pairs=config.include_zero_pairs,
        zero_pair_ratio=config.zero_pair_ratio,
    )
    fit_label = 'train pairs + sampled zeros' if config.include_zero_pairs else 'train nonzero pairs'
    print(f'  GMEL_GPS: fitting {config.decoder_type.upper()} on '
          f'{X_fit.shape[0]:,} {fit_label} ...')

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
    mf, mnz, mt = _print_stage_metrics("Tree Decoder", pred, od_np, test_mask)

    save_metrics_to_csv(run_id, run_name, config, mf, mnz, mt,
                        n_params, epoch, status)
    save_model_weights(run_id, model, config)

    return {
        'name': run_name,
        'model': model,
        'decoder': decoder,
        'config': config,
        'history': history,
        'loss_plot_path': str(saved_plot_path) if saved_plot_path is not None else None,
        'metrics_bilinear_full': bilinear_mf,
        'metrics_bilinear_nonzero': bilinear_mnz,
        'metrics_bilinear_test_pairs': bilinear_mt,
        'metrics_full': mf,
        'metrics_nonzero': mnz,
        'metrics_test_pairs': mt,
        'status': status,
    }
