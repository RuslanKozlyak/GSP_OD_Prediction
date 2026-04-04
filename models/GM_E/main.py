import time
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from models.shared.metrics import compute_metrics


def train(x_train, y_train, xs_valid, ys_valid, xs_valid_full=None, ys_valid_full=None,
          device=None, batch_size=50_000, max_epochs=10000, patience=100):
    """Train GRAVITY (GM_E) on pre-built feature arrays.

    Args:
        x_train: np.ndarray (N, F) — pair features (pop + distance)
        y_train: np.ndarray (N,)   — OD values
        xs_valid: list of np.ndarray — per-city validation features
        ys_valid: list of np.ndarray — per-city validation OD values
        xs_valid_full / ys_valid_full: optional full-matrix view used for
            CPC_full monitoring
        device: torch.device
        batch_size: DataLoader mini-batch size
        max_epochs / patience: training schedule

    Returns:
        predict: callable(x: np.ndarray) -> np.ndarray
    """
    import os, sys
    sys.modules.pop('model', None)  # prevent collision when run after another model
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import GRAVITY

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if xs_valid_full is None or ys_valid_full is None:
        xs_valid_full = xs_valid
        ys_valid_full = ys_valid

    # Filter zero OD pairs + log-space targets
    nz = y_train > 0
    x_nz, y_nz_log = x_train[nz], np.log1p(y_train[nz])

    ds = TensorDataset(
        torch.FloatTensor(x_nz),
        torch.FloatTensor(y_nz_log),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    del ds, x_nz, y_nz_log; gc.collect()

    # Pre-compute validation tensors once (avoid per-city alloc + GPU round-trip every epoch)
    _vx = [xv[yv > 0] for xv, yv in zip(xs_valid, ys_valid) if (yv > 0).any()]
    _vy = [np.log1p(yv[yv > 0]) for xv, yv in zip(xs_valid, ys_valid) if (yv > 0).any()]
    if _vx:
        xv_all_t = torch.FloatTensor(np.concatenate(_vx)).to(device)
        yv_log_all = np.concatenate(_vy)
    else:
        xv_all_t = None
    del _vx, _vy

    net = GRAVITY().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    def _predict_np(x_np):
        preds = []
        for start in range(0, x_np.shape[0], batch_size):
            xb = torch.FloatTensor(x_np[start:start + batch_size]).to(device)
            preds.append(np.atleast_1d(net(xb).squeeze().cpu().numpy()))
        if not preds:
            return np.empty((0,), dtype=np.float32)
        return np.maximum(np.concatenate(preds), 0.0)

    best_vl = np.inf
    best_pat = patience
    best_state = None
    val_cpc_vals = []
    val_cpc_fulls = []
    yv_log_all_t = torch.FloatTensor(yv_log_all).to(device) if xv_all_t is not None else None
    pbar = tqdm(range(max_epochs), desc='GM_E', unit='ep')
    for ep in pbar:
        net.train()
        ep_losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = torch.mean((torch.log1p(net(xb).squeeze()) - yb) ** 2)
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        net.eval()
        with torch.no_grad():
            if xv_all_t is not None:
                yh_all = net(xv_all_t).squeeze()
                vl = float(((torch.log1p(yh_all) - yv_log_all_t) ** 2).mean().item())
            else:
                vl = np.inf

            vc_vals = []
            for xv, yv in zip(xs_valid, ys_valid):
                pred_val = _predict_np(xv)
                vc_vals.append(compute_metrics(pred_val, yv)['CPC'])

            vcpcs = []
            for xv_full, yv_full in zip(xs_valid_full, ys_valid_full):
                pred_full = _predict_np(xv_full)
                vcpcs.append(compute_metrics(pred_full, yv_full)['CPC'])
            vc_val = float(np.mean(vc_vals)) if vc_vals else 0.0
            vc = float(np.mean(vcpcs)) if vcpcs else 0.0

        val_cpc_vals.append(vc_val)
        val_cpc_fulls.append(vc)
        pbar.set_postfix(
            loss=f'{np.mean(ep_losses):.4g}',
            val=f'{vl:.4g}',
            CPC_val=f'{vc_val:.4g}',
            CPC_full=f'{vc:.4g}',
            pat=best_pat,
        )

        if vl < best_vl:
            best_vl = vl
            best_pat = patience
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            best_pat -= 1
            if best_pat == 0:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    _net = net

    def predict(x):
        _net.eval()
        with torch.no_grad():
            return np.atleast_1d(_net(torch.FloatTensor(x).to(device)).squeeze().cpu().numpy())

    predict.val_cpc_vals = val_cpc_vals
    predict.val_cpc_fulls = val_cpc_fulls
    return predict
