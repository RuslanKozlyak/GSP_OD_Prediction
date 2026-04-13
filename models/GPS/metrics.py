"""GPS-specific inference and evaluation.

Standard metrics are in models.shared.metrics — imported here for backward compat.
"""
import numpy as np
import torch

from .config import DEST_BATCH_SIZE, device

# Re-export shared metrics for backward compatibility
from models.shared.metrics import (
    CPC, RMSE, MAE, MAPE, SMAPE, MSE, NRMSE,
    RMSE_nonzero, MAE_nonzero, MAPE_nonzero, SMAPE_nonzero, CPC_nonzero,
    MSE_nonzero, NRMSE_nonzero,
    accuracy, matrix_COS_similarity,
    JSD_inflow, JSD_outflow, JSD_ODflow,
    cal_od_metrics, compute_metrics, average_listed_metrics,
    canonical_od_metrics, citywise_segmented_metrics,
    masked_train_val_cpc_metrics, masked_split_metrics, num_regions,
)


def _softplus_np(x):
    x = np.clip(x, -60.0, 20.0)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def predict_full_matrix(model, cd, config, dbs=DEST_BATCH_SIZE):
    """Run GPS model encode+decode to produce full N×N OD prediction matrix."""
    model.eval()
    pm = config.prediction_mode
    use_log_flow = config.use_log_transform and config.loss_type in ('huber', 'multitask', 'mae')
    use_log_norm = use_log_flow and pm == 'normalized'
    is_gravity = getattr(config, 'decoder_type', None) == 'gravity_guided'
    nn_ = cd['num_nodes']
    of = cd['outflow_full']
    pred = np.zeros((nn_, nn_), dtype=np.float32)
    with torch.no_grad():
        ne = model.encode(cd['graph_data'])
        for oi in range(nn_):
            row = np.zeros(nn_, dtype=np.float32)
            for bs in range(0, nn_, dbs):
                be = min(bs + dbs, nn_)
                di = torch.LongTensor(np.arange(bs, be)).to(device)
                row[bs:be] = model.decode_row(
                    ne, oi, di, cd['distance_matrix'],
                    coords=cd.get('coords_tensor'),
                ).cpu().numpy()
            if config.loss_type == 'zinb':
                row = np.log1p(np.exp(row))
            elif pm == 'normalized':
                if use_log_norm:
                    prob = 1.0 / (1.0 + np.exp(-np.clip(row, -60, 60)))
                    row = np.expm1(prob * np.log1p(max(float(of[oi]), 0.0)))
                else:
                    row = np.exp(row - row.max())
                    row = row / (row.sum() + 1e-10) * of[oi]
            else:
                if is_gravity:
                    row = _softplus_np(row) if use_log_flow else np.exp(np.clip(row, -60.0, 20.0))
                else:
                    row = np.maximum(row, 0)
            if use_log_flow and pm != 'normalized':
                row = np.expm1(np.maximum(row, 0))
            pred[oi] = row
    return pred


def summarize_prediction_metrics(pred, cd, is_test_city=True):
    """Compute one consistent metric bundle for a predicted city matrix.

    This helper is the shared source of truth for GPS evaluation used by both
    training (`gps_od.ipynb` via models.GPS.main) and benchmarking
    (`benchmark.ipynb` via benchmarking.gps_loader).
    """
    od = cd['od_matrix_np']
    full_metrics = cal_od_metrics(pred, od)

    nz = od > 0
    if np.any(nz):
        nonzero_metrics = compute_metrics(pred[nz], od[nz].astype(float))
    else:
        nonzero_metrics = {'CPC': 0.0, 'MAE': 0.0, 'RMSE': 0.0}

    is_multi = cd.get('split_scope') == 'multi_city'
    test_mask = None if is_multi else cd.get('test_mask')
    test_full_mask = None if is_multi else cd.get('test_full_mask')
    combined_metrics = canonical_od_metrics(
        pred,
        od,
        test_mask=test_mask,
        test_full_mask=test_full_mask,
        train_mask=cd.get('train_mask'),
        val_mask=cd.get('val_mask'),
        train_full_mask=cd.get('train_full_mask'),
        val_full_mask=cd.get('val_full_mask'),
        is_test_city=is_test_city,
    )
    test_metrics = {
        'CPC': combined_metrics['CPC_test'],
        'MAE': combined_metrics['MAE_test'],
        'RMSE': combined_metrics['RMSE_test'],
    }

    return {
        'full': full_metrics,
        'nonzero': nonzero_metrics,
        'test': test_metrics,
        'combined': combined_metrics,
    }


def evaluate_full_matrix(model, cd, config, dest_batch_size=DEST_BATCH_SIZE,
                         is_test_city=True):
    """Predict full matrix and compute canonical metrics.

    Returns:
        pred:      N×N prediction matrix
        combined:  canonical metric dict from canonical_od_metrics
    """
    pred = predict_full_matrix(model, cd, config, dest_batch_size)
    summary = summarize_prediction_metrics(pred, cd, is_test_city=is_test_city)
    return pred, summary['combined']
