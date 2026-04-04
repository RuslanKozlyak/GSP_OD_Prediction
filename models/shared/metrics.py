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


def average_listed_metrics(listed_metrics):
    sums = defaultdict(float)
    for d in listed_metrics:
        for key, value in d.items():
            sums[key] += value
    averages = {key: value / len(listed_metrics) for key, value in sums.items()}
    return averages


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
