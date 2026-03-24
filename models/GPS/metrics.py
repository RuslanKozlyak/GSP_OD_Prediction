import math
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import entropy

from .config import DEST_BATCH_SIZE, device


# ─── GPS-specific metrics (used during training) ─────────────────────────────

def compute_cpc(p, t):
    d = p.sum() + t.sum()
    return 0.0 if d < 1e-10 else float(2 * np.minimum(p, t).sum() / d)


def compute_metrics(p, t):
    return {
        'CPC': compute_cpc(p, t),
        'MAE': float(np.abs(p - t).mean()),
        'RMSE': float(np.sqrt(((p - t) ** 2).mean())),
    }


def predict_full_matrix(model, cd, config, dbs=DEST_BATCH_SIZE):
    model.eval()
    pm = config.prediction_mode
    ul = config.use_log_transform
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
                row[bs:be] = model.decode_row(ne, oi, di, cd['distance_matrix']).cpu().numpy()
            if config.loss_type == 'zinb':
                row = np.log1p(np.exp(row))
            elif pm == 'normalized':
                row = np.exp(row - row.max())
                row = row / (row.sum() + 1e-10) * of[oi]
            else:
                row = np.maximum(row, 0)
            if ul and config.loss_type not in ('ce', 'zinb'):
                row = np.expm1(np.maximum(row, 0))
            pred[oi] = row
    return pred


def evaluate_full_matrix(model, cd, config, dest_batch_size=DEST_BATCH_SIZE):
    pred = predict_full_matrix(model, cd, config, dest_batch_size)
    od = cd['od_matrix_np']
    nz = od > 0
    return (
        pred,
        compute_metrics(pred.ravel(), od.ravel().astype(float)),
        compute_metrics(pred[nz], od[nz].astype(float)),
    )


# ─── Standard metrics suite (identical to models/DGM/metrics.py) ─────────────
# Used for benchmark comparison with other models

def RMSE(a, b):
    if type(a) == type(np.array([1, 1])):
        return np.sqrt(((a - b) ** 2).mean())
    else:
        return ((a - b) ** 2).mean().sqrt()


def NRMSE(a, b):
    return RMSE(a, b) / b.std()


def MAE(a, b):
    if type(a) == type(np.array([1, 1])):
        return np.abs(a - b).mean()
    else:
        return (a - b).abs().mean()


def MAPE(a, b):
    if type(a) == type(np.array([1, 1])):
        return (np.abs(a - b) / (np.abs(b) + 1)).mean()
    else:
        return ((a - b).abs() / (b.abs() + 1)).mean()


def MSE(a, b):
    return ((a - b) ** 2).mean()


def SMAPE(a, b):
    if type(a) == type(np.array([1, 1])):
        return (np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2 + 1e-20)).mean()
    else:
        return ((a - b).abs() / ((a.abs() + b.abs()) / 2 + 1e-20)).mean()


def CPC(a, b):
    if ((a < 0).sum() + (b < 0).sum()) > 0:
        raise ValueError("OD flow should not be less than zero.")
    mn = np.minimum(a, b)
    return 2 * mn.sum() / (a.sum() + b.sum())


def RMSE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return RMSE(a, b)


def MSE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return MSE(a, b)


def MAE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return MAE(a, b)


def MAPE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return MAPE(a, b)


def SMAPE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return SMAPE(a, b)


def CPC_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return CPC(a, b)


def accuracy(a, b):
    a = a.copy()
    a[a < 1] = 0
    idx_a = a.nonzero()
    idx_b = b.nonzero()
    a = np.zeros(a.shape)
    a[idx_a] = 1
    b_bin = np.zeros(b.shape)
    b_bin[idx_b] = 1
    sim = (a == b_bin).sum() / (a.shape[0] ** 2)
    return sim


def matrix_COS_similarity(a, b):
    if type(a) == type(np.array([1, 1])):
        a_row_norm = np.sqrt((a ** 2).sum(0))
        b_row_norm = np.sqrt((b ** 2).sum(0))
    else:
        a_row_norm = (a ** 2).sum(0).sqrt()
        b_row_norm = (b ** 2).sum(0).sqrt()
    row_sim = (a * b).sum(0) / (a_row_norm * b_row_norm + 1e-20)
    if type(a) == type(np.array([1, 1])):
        a_col_norm = np.sqrt((a ** 2).sum(1))
        b_col_norm = np.sqrt((b ** 2).sum(1))
    else:
        a_col_norm = (a ** 2).sum(1).sqrt()
        b_col_norm = (b ** 2).sum(1).sqrt()
    col_sim = (a * b).sum(1) / (a_col_norm * b_col_norm + 1e-20)
    final_sim = (row_sim.sum() + col_sim.sum()) / (row_sim.shape[0] * 2)
    return final_sim


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


def JSD_inflow(a, b):
    a_in, b_in = a.sum(0), b.sum(0)
    sections, b_dist = values_to_bucket(b_in)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i + 1]
        frequency = np.sum((a_in >= low) & (a_in < high))
        a_dist.append(frequency)
    a_dist = np.array(a_dist) / np.array(a_dist).sum()
    b_dist = np.array(b_dist) / np.array(b_dist).sum()
    return JS_divergence(a_dist, b_dist)


def JSD_outflow(a, b):
    a_out, b_out = a.sum(1), b.sum(1)
    sections, b_dist = values_to_bucket(b_out)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i + 1]
        frequency = np.sum((a_out >= low) & (a_out < high))
        a_dist.append(frequency)
    a_dist = np.array(a_dist) / np.array(a_dist).sum()
    b_dist = np.array(b_dist) / np.array(b_dist).sum()
    return JS_divergence(a_dist, b_dist)


def JSD_ODflow(a, b):
    a, b = a.reshape([-1]), b.reshape([-1])
    sections, b_dist = values_to_bucket(b)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i + 1]
        frequency = np.sum((a >= low) & (a < high))
        a_dist.append(frequency)
    a_dist = np.array(a_dist) / np.array(a_dist).sum()
    b_dist = np.array(b_dist) / np.array(b_dist).sum()
    return JS_divergence(a_dist, b_dist)


def num_regions(a, b):
    return b.shape[0]


def cal_od_metrics(a, b):
    """Full metrics suite. b must be groundtruth."""
    metrics = {
        "num_regions": num_regions(a, b),
        "RMSE": RMSE(a, b).item() if hasattr(RMSE(a, b), 'item') else float(RMSE(a, b)),
        "NRMSE": NRMSE(a, b).item() if hasattr(NRMSE(a, b), 'item') else float(NRMSE(a, b)),
        "MAE": MAE(a, b).item() if hasattr(MAE(a, b), 'item') else float(MAE(a, b)),
        "MAPE": MAPE(a, b).item() if hasattr(MAPE(a, b), 'item') else float(MAPE(a, b)),
        "SMAPE": SMAPE(a, b).item() if hasattr(SMAPE(a, b), 'item') else float(SMAPE(a, b)),
        "CPC": CPC(a, b).item() if hasattr(CPC(a, b), 'item') else float(CPC(a, b)),
        "RMSE_nonzero": RMSE_nonzero(a, b).item() if hasattr(RMSE_nonzero(a, b), 'item') else float(RMSE_nonzero(a, b)),
        "MAE_nonzero": MAE_nonzero(a, b).item() if hasattr(MAE_nonzero(a, b), 'item') else float(MAE_nonzero(a, b)),
        "MAPE_nonzero": MAPE_nonzero(a, b).item() if hasattr(MAPE_nonzero(a, b), 'item') else float(MAPE_nonzero(a, b)),
        "SMAPE_nonzero": SMAPE_nonzero(a, b).item() if hasattr(SMAPE_nonzero(a, b), 'item') else float(SMAPE_nonzero(a, b)),
        "CPC_nonzero": CPC_nonzero(a, b).item() if hasattr(CPC_nonzero(a, b), 'item') else float(CPC_nonzero(a, b)),
        "accuracy": accuracy(a, b).item() if hasattr(accuracy(a, b), 'item') else float(accuracy(a, b)),
        "matrix_COS_similarity": matrix_COS_similarity(a, b).item() if hasattr(matrix_COS_similarity(a, b), 'item') else float(matrix_COS_similarity(a, b)),
        "JSD_inflow": JSD_inflow(a, b).item() if hasattr(JSD_inflow(a, b), 'item') else float(JSD_inflow(a, b)),
        "JSD_outflow": JSD_outflow(a, b).item() if hasattr(JSD_outflow(a, b), 'item') else float(JSD_outflow(a, b)),
        "JSD_ODflow": JSD_ODflow(a, b).item() if hasattr(JSD_ODflow(a, b), 'item') else float(JSD_ODflow(a, b)),
    }
    return metrics


def average_listed_metrics(listed_metrics):
    sums = defaultdict(float)
    for d in listed_metrics:
        for key, value in d.items():
            sums[key] += value
    averages = {key: value / len(listed_metrics) for key, value in sums.items()}
    return averages
