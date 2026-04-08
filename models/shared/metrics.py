"""Canonical OD metrics module.

All models (GPS, baselines, graph models) must use these functions
for fair comparison. This is the single source of truth.
"""
from collections import defaultdict

import numpy as np
from scipy.stats import entropy


def _safe_scalar(x):
    """Convert numpy/torch scalar to Python float safely."""
    return x.item() if hasattr(x, 'item') else float(x)


# ─── Base metrics ────────────────────────────────────────────────────────────

def RMSE(a, b):
    if isinstance(a, np.ndarray):
        return np.sqrt(((a - b) ** 2).mean())
    else:
        return ((a - b) ** 2).mean().sqrt()


def NRMSE(a, b):
    return RMSE(a, b) / b.std()


def MAE(a, b):
    if isinstance(a, np.ndarray):
        return np.abs(a - b).mean()
    else:
        return (a - b).abs().mean()


def MAPE(a, b):
    if isinstance(a, np.ndarray):
        return (np.abs(a - b) / (np.abs(b) + 1)).mean()
    else:
        return ((a - b).abs() / (b.abs() + 1)).mean()


def MSE(a, b):
    return ((a - b) ** 2).mean()


def SMAPE(a, b):
    if isinstance(a, np.ndarray):
        return (np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2 + 1e-20)).mean()
    else:
        return ((a - b).abs() / ((a.abs() + b.abs()) / 2 + 1e-20)).mean()


def CPC(a, b):
    if ((a < 0).sum() + (b < 0).sum()) > 0:
        raise ValueError("OD flow should not be less than zero.")
    mn = np.minimum(a, b) if isinstance(a, np.ndarray) else __import__('torch').minimum(a, b)
    return 2 * mn.sum() / (a.sum() + b.sum())


# ─── Nonzero variants ────────────────────────────────────────────────────────

def _select_nonzero(a, b):
    if isinstance(a, np.ndarray):
        idx = b.nonzero()
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
    return a[idx], b[idx]


def RMSE_nonzero(a, b):
    return RMSE(*_select_nonzero(a, b))

def MSE_nonzero(a, b):
    return MSE(*_select_nonzero(a, b))

def NRMSE_nonzero(a, b):
    return NRMSE(*_select_nonzero(a, b))

def MAE_nonzero(a, b):
    return MAE(*_select_nonzero(a, b))

def MAPE_nonzero(a, b):
    return MAPE(*_select_nonzero(a, b))

def SMAPE_nonzero(a, b):
    return SMAPE(*_select_nonzero(a, b))

def CPC_nonzero(a, b):
    return CPC(*_select_nonzero(a, b))


# ─── Structure metrics ───────────────────────────────────────────────────────

def accuracy(a, b):
    a = a.copy() if isinstance(a, np.ndarray) else a.clone()
    if isinstance(a, np.ndarray):
        a[a < 1] = 0
        idx_a = a.nonzero()
        idx_b = b.nonzero()
        a = np.zeros(a.shape)
        a[idx_a] = 1
        b_bin = np.zeros(b.shape)
        b_bin[idx_b] = 1
    else:
        import torch
        a[a < 1] = 0
        idx_a = a.nonzero()
        idx_b = b.nonzero()
        idx_a = (idx_a[:, 0], idx_a[:, 1])
        idx_b = (idx_b[:, 0], idx_b[:, 1])
        a = torch.zeros_like(a)
        a[idx_a] = 1
        b_bin = torch.zeros_like(b)
        b_bin[idx_b] = 1
    sim = (a == b_bin).sum() / (a.shape[0] ** 2)
    return sim


def matrix_COS_similarity(a, b):
    if isinstance(a, np.ndarray):
        a_row_norm = np.sqrt((a ** 2).sum(0))
        b_row_norm = np.sqrt((b ** 2).sum(0))
    else:
        a_row_norm = (a ** 2).sum(0).sqrt()
        b_row_norm = (b ** 2).sum(0).sqrt()
    row_sim = (a * b).sum(0) / (a_row_norm * b_row_norm + 1e-20)

    if isinstance(a, np.ndarray):
        a_col_norm = np.sqrt((a ** 2).sum(1))
        b_col_norm = np.sqrt((b ** 2).sum(1))
    else:
        a_col_norm = (a ** 2).sum(1).sqrt()
        b_col_norm = (b ** 2).sum(1).sqrt()
    col_sim = (a * b).sum(1) / (a_col_norm * b_col_norm + 1e-20)

    final_sim = (row_sim.sum() + col_sim.sum()) / (row_sim.shape[0] * 2)
    return final_sim


# ─── Distribution metrics ────────────────────────────────────────────────────

def values_to_bucket(values):
    max_ = values.max()
    i = 0
    leftright = []
    nums = []
    while True:
        if i == 0:
            left = 0
            right = 1
            leftright.append(left)
            leftright.append(right)
            i += 1
        else:
            left = i
            right = i * 2
            leftright.append(right)
            i = i * 2
        nums.append(((values >= left) & (values < right)).sum())
        if right > max_:
            break
    return leftright, nums


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * entropy(p, M, base=2) + 0.5 * entropy(q, M, base=2)


def _jsd_flow(a_flow, b_flow):
    sections, b_dist = values_to_bucket(b_flow)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i + 1]
        frequency = np.sum((a_flow >= low) & (a_flow < high))
        a_dist.append(frequency)
    a_dist = np.array(a_dist) / np.array(a_dist).sum()
    b_dist = np.array(b_dist) / np.array(b_dist).sum()
    return JS_divergence(a_dist, b_dist)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return x.cpu().numpy()


def JSD_inflow(a, b):
    return _jsd_flow(_to_numpy(a).sum(0), _to_numpy(b).sum(0))

def JSD_outflow(a, b):
    return _jsd_flow(_to_numpy(a).sum(1), _to_numpy(b).sum(1))

def JSD_ODflow(a, b):
    return _jsd_flow(_to_numpy(a).reshape(-1), _to_numpy(b).reshape(-1))

# Aliases
JSD_in = JSD_inflow
JSD_out = JSD_outflow
JSD_indegree = JSD_inflow
JSD_outdegree = JSD_outflow


# ─── Aggregation ─────────────────────────────────────────────────────────────

def num_regions(a, b):
    return b.shape[0]


def cal_od_metrics(a, b):
    """Full metrics suite. b must be groundtruth."""
    metrics = {
        "num_regions": num_regions(a, b),
        "RMSE": _safe_scalar(RMSE(a, b)),
        "NRMSE": _safe_scalar(NRMSE(a, b)),
        "MAE": _safe_scalar(MAE(a, b)),
        "MAPE": _safe_scalar(MAPE(a, b)),
        "SMAPE": _safe_scalar(SMAPE(a, b)),
        "CPC": _safe_scalar(CPC(a, b)),
        "RMSE_nonzero": _safe_scalar(RMSE_nonzero(a, b)),
        "MAE_nonzero": _safe_scalar(MAE_nonzero(a, b)),
        "MAPE_nonzero": _safe_scalar(MAPE_nonzero(a, b)),
        "SMAPE_nonzero": _safe_scalar(SMAPE_nonzero(a, b)),
        "CPC_nonzero": _safe_scalar(CPC_nonzero(a, b)),
        "accuracy": _safe_scalar(accuracy(a, b)),
        "matrix_COS_similarity": _safe_scalar(matrix_COS_similarity(a, b)),
        "JSD_inflow": _safe_scalar(JSD_inflow(a, b)),
        "JSD_outflow": _safe_scalar(JSD_outflow(a, b)),
        "JSD_ODflow": _safe_scalar(JSD_ODflow(a, b)),
    }
    return metrics


def compute_metrics(p, t):
    """Quick 3-metric summary for training loop monitoring."""
    return {
        'CPC': _safe_scalar(CPC(p, t)) if (p.sum() + t.sum()) > 1e-10 else 0.0,
        'MAE': _safe_scalar(MAE(p, t)),
        'RMSE': _safe_scalar(RMSE(p, t)),
    }


def _nan_quick_metrics():
    return {'CPC': float('nan'), 'MAE': float('nan'), 'RMSE': float('nan')}


def canonical_od_metrics(
    pred_matrix,
    od_matrix,
    *,
    test_mask=None,
    train_mask=None,
    val_mask=None,
    train_full_mask=None,
    val_full_mask=None,
    is_test_city=True,
):
    """Return the canonical benchmark metric schema for one OD matrix.

    Canonical keys:
    - *_full: metrics on the full OD matrix.
    - *_nz: metrics on nonzero target OD pairs.
    - *_test: metrics on the test split, or full matrix for test cities.
    - CPC_train_*/CPC_val_*: split CPC diagnostics when masks are provided.
    """
    pred_matrix = np.asarray(pred_matrix)
    od_matrix = np.asarray(od_matrix)
    full_metrics = cal_od_metrics(pred_matrix, od_matrix)

    nz_mask = od_matrix > 0
    nonzero_metrics = (
        compute_metrics(pred_matrix[nz_mask], od_matrix[nz_mask].astype(float))
        if np.any(nz_mask) else
        {'CPC': 0.0, 'MAE': 0.0, 'RMSE': 0.0}
    )

    if test_mask is not None and np.any(test_mask):
        test_mask = np.asarray(test_mask, dtype=bool)
        test_metrics = compute_metrics(
            pred_matrix[test_mask], od_matrix[test_mask].astype(float)
        )
    elif is_test_city:
        test_metrics = {
            'CPC': full_metrics['CPC'],
            'MAE': full_metrics['MAE'],
            'RMSE': full_metrics['RMSE'],
        }
    else:
        test_metrics = _nan_quick_metrics()

    metrics = {
        'num_regions': full_metrics['num_regions'],
        'CPC_full': full_metrics['CPC'],
        'MAE_full': full_metrics['MAE'],
        'RMSE_full': full_metrics['RMSE'],
        'NRMSE_full': full_metrics['NRMSE'],
        'MAPE_full': full_metrics['MAPE'],
        'SMAPE_full': full_metrics['SMAPE'],
        'CPC_nz': nonzero_metrics['CPC'],
        'MAE_nz': nonzero_metrics['MAE'],
        'RMSE_nz': nonzero_metrics['RMSE'],
        'MAPE_nz': full_metrics['MAPE_nonzero'],
        'SMAPE_nz': full_metrics['SMAPE_nonzero'],
        'CPC_test': test_metrics['CPC'],
        'MAE_test': test_metrics['MAE'],
        'RMSE_test': test_metrics['RMSE'],
        'accuracy': full_metrics['accuracy'],
        'matrix_COS_similarity': full_metrics['matrix_COS_similarity'],
        'JSD_inflow': full_metrics['JSD_inflow'],
        'JSD_outflow': full_metrics['JSD_outflow'],
        'JSD_ODflow': full_metrics['JSD_ODflow'],
    }

    if train_mask is not None and val_mask is not None:
        metrics.update(
            masked_train_val_cpc_metrics(
                pred_matrix,
                od_matrix,
                train_mask,
                val_mask,
                train_full_mask=train_full_mask,
                val_full_mask=val_full_mask,
            )
        )
        metrics['CPC_val'] = metrics['CPC_val_nz']
    elif not np.isnan(metrics['CPC_test']):
        metrics['CPC_val'] = float('nan')

    return metrics


def cpc_full(pred, target):
    """CPC on flattened full matrices."""
    return compute_metrics(
        np.asarray(pred).reshape(-1),
        np.asarray(target).reshape(-1),
    )['CPC']


def cpc_nonzero(pred_matrix, target_matrix, mask=None):
    """CPC on nonzero target entries or an explicit boolean mask."""
    pred_matrix = np.asarray(pred_matrix)
    target_matrix = np.asarray(target_matrix)
    mask = target_matrix > 0 if mask is None else np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    return compute_metrics(pred_matrix[mask], target_matrix[mask].astype(float))['CPC']


def masked_train_val_cpc_metrics(
    pred_matrix,
    od_matrix,
    train_mask,
    val_mask,
    train_full_mask=None,
    val_full_mask=None,
):
    """Common train/val CPC metrics for single-city pair masks.

    CPC_*_full compares predicted/target split matrices after applying full
    pair masks, including zeros when those masks are provided. CPC_*_nz is
    restricted to the nonzero split mask entries.
    """
    pred_matrix = np.asarray(pred_matrix)
    od_matrix = np.asarray(od_matrix)
    train_mask = np.asarray(train_mask, dtype=bool)
    val_mask = np.asarray(val_mask, dtype=bool)
    train_full_mask = (
        train_mask if train_full_mask is None
        else np.asarray(train_full_mask, dtype=bool)
    )
    val_full_mask = (
        val_mask if val_full_mask is None
        else np.asarray(val_full_mask, dtype=bool)
    )
    return {
        'CPC_train_full': cpc_full(
            pred_matrix * train_full_mask, od_matrix * train_full_mask
        ),
        'CPC_val_full': cpc_full(
            pred_matrix * val_full_mask, od_matrix * val_full_mask
        ),
        'CPC_train_nz': cpc_nonzero(pred_matrix, od_matrix, train_mask),
        'CPC_val_nz': cpc_nonzero(pred_matrix, od_matrix, val_mask),
    }


def average_matrix_cpc_metrics(pred_matrices, od_matrices, prefix):
    """Average full/nonzero CPC over a list of city matrices."""
    cpc_fulls = []
    cpc_nzs = []
    for pred_matrix, od_matrix in zip(pred_matrices, od_matrices):
        cpc_fulls.append(cpc_full(pred_matrix, od_matrix))
        cpc_nzs.append(cpc_nonzero(pred_matrix, od_matrix))
    return {
        f'CPC_{prefix}_full': float(np.mean(cpc_fulls)) if cpc_fulls else float('nan'),
        f'CPC_{prefix}_nz': float(np.mean(cpc_nzs)) if cpc_nzs else float('nan'),
    }


def format_train_val_cpc_metrics(metrics):
    return (
        f"CPC_train_full={metrics['CPC_train_full']:.4f}  "
        f"CPC_val_full={metrics['CPC_val_full']:.4f}  "
        f"CPC_train_nz={metrics['CPC_train_nz']:.4f}  "
        f"CPC_val_nz={metrics['CPC_val_nz']:.4f}"
    )


def average_listed_metrics(listed_metrics):
    if not listed_metrics:
        return {}
    keys = sorted({
        key
        for d in listed_metrics
        for key, value in d.items()
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
    })
    return {
        key: float(np.mean([
            d[key] for d in listed_metrics
            if key in d and isinstance(d[key], (int, float, np.number))
        ]))
        for key in keys
    }


def citywise_segmented_metrics(valid_metrics):
    SEG_metrics = {
        "(0, 10]": [], "(10, 50]": [], "(50, 100]": [],
        "(100, 200]": [], "(200, 500]": [], "(500, 1000]": [],
        "(1000, 2000]": [], "(2000, +inf]": [],
    }
    boundaries = [10, 50, 100, 200, 500, 1000, 2000, float('inf')]
    keys = list(SEG_metrics.keys())

    for item in valid_metrics:
        nr = item["num_regions"]
        for i, upper in enumerate(boundaries):
            lower = boundaries[i - 1] if i > 0 else 0
            if lower < nr <= upper:
                SEG_metrics[keys[i]].append(item)
                break

    for key in keys:
        if SEG_metrics[key]:
            SEG_metrics[key] = average_listed_metrics(SEG_metrics[key])

    return SEG_metrics


# ─── Extra metrics from WeDAN/DiffODGen ─────────────────────────────────────

def false_negative_rate(a, b):
    a, b = _to_numpy(a), _to_numpy(b)
    return np.sum((a == 0) & (b == 1)) / np.sum(b == 1)

def false_positive_rate(a, b):
    a, b = _to_numpy(a), _to_numpy(b)
    if np.sum(b == 0) != 0:
        return np.sum((a == 1) & (b == 0)) / np.sum(b == 0)
    return np.float32(np.nan)

def nonzero_flow_fraction(a, b):
    a, b = _to_numpy(a), _to_numpy(b)
    a_frac = (a == 1).sum() / (a.shape[0] ** 2)
    b_frac = (b == 1).sum() / (b.shape[0] ** 2)
    return np.float32(np.abs(a_frac - b_frac) / b_frac)
