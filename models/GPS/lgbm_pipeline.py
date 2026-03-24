import csv
from datetime import datetime

import numpy as np
import torch
import lightgbm as lgb

from .config import TrainingConfig, METRICS_CSV, device, ensure_dirs
from .metrics import compute_metrics


def train_lgbm_from_model(run_id, city_data, donor_model, donor_name):
    """Train LightGBM on GPS embeddings from donor_model."""
    print(f"\n{'='*70}\n  LGBM: {run_id} (donor: {donor_name})\n{'='*70}")
    nn_ = city_data['num_nodes']
    od = city_data['od_matrix_np']
    tm = city_data['train_mask']
    vm = city_data['val_mask']
    tsm = city_data['test_mask']
    nfs = city_data['node_features_scaled']
    ds = city_data['distances_scaled']

    donor_model.eval()
    with torch.no_grad():
        embs = donor_model.encode(city_data['graph_data']).cpu().numpy()
    ed = embs.shape[1]
    nfd = nfs.shape[1]

    def build_features(mask):
        oi, di = np.where(mask)
        n = len(oi)
        feat = np.zeros((n, ed * 2 + 1 + nfd * 2), dtype=np.float32)
        feat[:, :ed] = embs[oi]
        feat[:, ed:2*ed] = embs[di]
        feat[:, 2*ed] = ds[oi, di]
        feat[:, 2*ed+1:2*ed+1+nfd] = nfs[oi]
        feat[:, 2*ed+1+nfd:] = nfs[di]
        return feat, od[oi, di].astype(float), oi, di

    X_train, y_train, _, _ = build_features(tm)
    X_val, y_val, _, _ = build_features(vm)
    print(f"  Train: {len(y_train):,} pairs, Val: {len(y_val):,} pairs")

    params = {
        'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.05,
        'num_leaves': 63, 'max_depth': 8, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'verbose': -1, 'seed': 42,
    }
    lgbm_model = lgb.train(
        params, lgb.Dataset(X_train, y_train), num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
    )

    # Predict full matrix
    print(f"  Predicting full {nn_}x{nn_} matrix...")
    full_mask = np.ones((nn_, nn_), bool)
    X_full, _, ao, ad = build_features(full_mask)
    pf = np.maximum(lgbm_model.predict(X_full), 0)
    pred = np.zeros((nn_, nn_), dtype=np.float32)
    pred[ao, ad] = pf.astype(np.float32)

    mf = compute_metrics(pred.ravel(), od.ravel().astype(float))
    nzm = od > 0
    mnz = compute_metrics(pred[nzm], od[nzm].astype(float))
    mt = compute_metrics(pred[tsm], od[tsm].astype(float))

    print(f"  Full:    CPC={mf['CPC']:.4f}  MAE={mf['MAE']:.4f}")
    print(f"  Nonzero: CPC={mnz['CPC']:.4f}")
    print(f"  Test:    CPC={mt['CPC']:.4f}")

    # Save to CSV
    ensure_dirs()
    row = {
        'timestamp': datetime.now().isoformat(), 'run_id': run_id,
        'name': f'LGBM({donor_name})', 'status': 'ok',
        'decoder': 'lgbm', 'loss_type': 'mae',
        'prediction_mode': 'raw', 'pe_type': '-', 'gps_norm_type': '-',
        'use_log_transform': False, 'n_params': 0,
        'epochs_trained': lgbm_model.best_iteration,
        'CPC_full': mf['CPC'], 'CPC_nz': mnz['CPC'], 'CPC_test': mt['CPC'],
        'MAE_full': mf['MAE'], 'RMSE_full': mf['RMSE'],
    }
    with open(METRICS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writerow(row)

    return {
        'name': f'LGBM({donor_name})', 'model': lgbm_model,
        'metrics_full': mf, 'metrics_nonzero': mnz, 'metrics_test_pairs': mt,
        'pred_matrix': pred,
        'config': TrainingConfig(decoder_type='lgbm', loss_type='mae'),
        'status': 'ok',
    }
