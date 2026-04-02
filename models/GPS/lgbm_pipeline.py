import csv
import json
from datetime import datetime

import numpy as np
import torch
import lightgbm as lgb

from .config import TrainingConfig, METRICS_CSV, WEIGHTS_DIR, device, ensure_dirs
from .metrics import compute_metrics, cal_od_metrics


def build_lgbm_features(embs, nfs, ds, od, mask):
    """Build LGBM feature matrix for OD pairs selected by mask.

    Features per pair: [origin_emb | dest_emb | distance | origin_nf | dest_nf]
    Total dim: ed*2 + 1 + nfd*2

    Args:
        embs: (N, ed) GPS encoder embeddings
        nfs:  (N, nfd) scaled node features
        ds:   (N, N) scaled distance matrix
        od:   (N, N) OD flow matrix (used only for y values)
        mask: (N, N) bool mask selecting which pairs to include

    Returns:
        feat: (M, ed*2+1+nfd*2) float32 feature matrix
        y:    (M,) float64 OD values
        oi:   (M,) origin indices
        di:   (M,) destination indices
    """
    oi, di = np.where(mask)
    ed, nfd = embs.shape[1], nfs.shape[1]
    feat = np.zeros((len(oi), ed * 2 + 1 + nfd * 2), dtype=np.float32)
    feat[:, :ed] = embs[oi]
    feat[:, ed:2*ed] = embs[di]
    feat[:, 2*ed] = ds[oi, di]
    feat[:, 2*ed+1:2*ed+1+nfd] = nfs[oi]
    feat[:, 2*ed+1+nfd:] = nfs[di]
    return feat, od[oi, di].astype(float), oi, di


def save_lgbm_model(run_id, lgbm_model, donor_id):
    """Save LightGBM model and donor metadata to WEIGHTS_DIR."""
    ensure_dirs()
    lgbm_model.save_model(str(WEIGHTS_DIR / f"{run_id}.lgbm"))
    (WEIGHTS_DIR / f"{run_id}_meta.json").write_text(
        json.dumps({"donor_id": donor_id})
    )
    print(f"  Saved: {WEIGHTS_DIR / run_id}.lgbm + _meta.json")


def load_lgbm_results(run_id, city_data):
    """Load a saved LGBM model and evaluate on city_data. Returns metrics dict or None.

    Requires:
        {run_id}.lgbm          — saved LightGBM booster
        {run_id}_meta.json     — {"donor_id": "..."} identifying the GPS encoder
        {donor_id}.pt          — GPS encoder weights
        {donor_id}.json        — GPS encoder config
    """
    from .config import load_model_config
    from .model import make_model

    lgbm_path = WEIGHTS_DIR / f"{run_id}.lgbm"
    meta_path = WEIGHTS_DIR / f"{run_id}_meta.json"

    if not lgbm_path.exists():
        print(f"  [SKIP] {run_id}: {lgbm_path.name} not found — run lgbm_od.ipynb first")
        return None

    # Resolve donor GPS run_id
    if meta_path.exists():
        donor_id = json.loads(meta_path.read_text())["donor_id"]
    else:
        # Fallback: infer by convention {donor_id}_lgbm
        donor_id = run_id[:-5] if run_id.endswith("_lgbm") else run_id

    donor_weight = WEIGHTS_DIR / f"{donor_id}.pt"
    donor_cfg = load_model_config(donor_id)
    if not donor_weight.exists() or donor_cfg is None:
        print(f"  [SKIP] {run_id}: GPS donor '{donor_id}' weights/config not found")
        return None

    print(f"  Loading {run_id} (donor: {donor_id}) ...")
    try:
        # Load GPS encoder
        gps_model = make_model(donor_cfg, graph_data_ref=city_data["graph_data"])
        gps_model.load_state_dict(torch.load(str(donor_weight), map_location=device))
        gps_model.to(device).eval()

        # Extract frozen embeddings
        with torch.no_grad():
            embs = gps_model.encode(city_data["graph_data"]).cpu().numpy()

        # Load LGBM booster
        lgbm_model = lgb.Booster(model_file=str(lgbm_path))

        # Predict full matrix
        nn_ = city_data["num_nodes"]
        od = city_data["od_matrix_np"]
        nfs = city_data["node_features_scaled"]
        ds = city_data["distances_scaled"]

        full_mask = np.ones((nn_, nn_), bool)
        X_full, _, ao, ad = build_lgbm_features(embs, nfs, ds, od, full_mask)
        pf = np.maximum(lgbm_model.predict(X_full), 0)
        pred = np.zeros((nn_, nn_), dtype=np.float32)
        pred[ao, ad] = pf.astype(np.float32)

        metrics = cal_od_metrics(pred, od)
        print(f"  {run_id}: CPC={metrics['CPC']:.4f}  MAE={metrics['MAE']:.4f}")
        return metrics

    except Exception as e:
        print(f"  ERROR loading {run_id}: {e}")
        return None
    finally:
        try:
            del gps_model
        except NameError:
            pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


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

    X_train, y_train, _, _ = build_lgbm_features(embs, nfs, ds, od, tm)
    X_val, y_val, _, _ = build_lgbm_features(embs, nfs, ds, od, vm)
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
    X_full, _, ao, ad = build_lgbm_features(embs, nfs, ds, od, full_mask)
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

    # Save model to disk
    save_lgbm_model(run_id, lgbm_model, donor_name)

    # Save to metrics CSV
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
