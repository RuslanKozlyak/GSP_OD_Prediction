import time
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from models.shared.metrics import (
    compute_metrics,
    masked_train_val_cpc_metrics,
    average_matrix_split_metrics,
    format_train_val_cpc_metrics,
)
from models.shared.plotting import save_loss_plot


def _clone_state_dict(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _make_predict_fn(net, feat_scaler, od_scaler, device, batch_size):
    def predict(x):
        net.eval()
        with torch.no_grad():
            y_log = od_scaler.renormalize(
                net(torch.FloatTensor(feat_scaler.transform(x)).to(device)).squeeze().cpu().numpy()
            )
            return np.atleast_1d(np.expm1(np.maximum(y_log, 0.0)))

    predict._serialization = {
        'input_dim': net.linear_in.in_features,
        'state_dict': _clone_state_dict(net),
        'feat_scaler': feat_scaler,
        'od_scaler_min': float(od_scaler.min_),
        'od_scaler_max': float(od_scaler.max_),
        'batch_size': int(batch_size),
    }
    return predict


def train(x_train, y_train, xs_valid, ys_valid, xs_valid_full=None, ys_valid_full=None,
          device=None, batch_size=50_000, max_epochs=300, patience=100,
          loss_plot_path=None, lr=3e-4, grad_clip=1.0, verbose=1,
          x_full=None, od_matrix=None, train_mask=None, val_mask=None,
          train_full_mask=None, val_full_mask=None,
          xs_train_full=None, ys_train_full=None):
    """Train DeepGravity on pre-built feature arrays.

    Args:
        x_train: np.ndarray (N, F) — concatenated pair features for training
        y_train: np.ndarray (N,)   — OD values for training
        xs_valid: list of np.ndarray — per-city validation features
        ys_valid: list of np.ndarray — per-city validation OD values
        xs_valid_full / ys_valid_full: optional full-matrix view used for
            CPC_full monitoring
        device: torch.device (defaults to cuda if available)
        batch_size: DataLoader mini-batch size
        max_epochs: max training epochs
        patience: early-stopping patience
        verbose: enables tqdm and training log messages

    Returns:
        predict: callable(x: np.ndarray) -> np.ndarray
    """
    import os, sys
    sys.modules.pop('model', None)  # prevent collision when run after another model
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import DeepGravity, OD_normer

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if xs_valid_full is None or ys_valid_full is None:
        xs_valid_full = xs_valid
        ys_valid_full = ys_valid

    single_city_split_ready = (
        x_full is not None
        and od_matrix is not None
        and train_mask is not None
        and val_mask is not None
    )
    multi_city_split_ready = (
        not single_city_split_ready
        and xs_train_full is not None
        and ys_train_full is not None
        and xs_valid_full is not None
        and ys_valid_full is not None
        and len(xs_train_full) > 0
        and len(xs_valid_full) > 0
    )

    # Filter zero OD pairs — avoids mode collapse on sparse matrices
    nz = y_train > 0
    x_nz, y_nz = x_train[nz], y_train[nz]

    feat_scaler = MinMaxScaler((-1, 1)).fit(x_nz)
    _y_log = np.log1p(y_nz)
    od_scaler = OD_normer(_y_log.min(), _y_log.max())

    ds = TensorDataset(
        torch.FloatTensor(feat_scaler.transform(x_nz)),
        torch.FloatTensor(od_scaler.normalize(_y_log)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    input_dim = x_train.shape[1]
    del ds, x_nz, y_nz, _y_log; gc.collect()
    net = DeepGravity(input_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def _predict_np(x_np):
        preds = []
        for start in range(0, x_np.shape[0], batch_size):
            xb = torch.FloatTensor(
                feat_scaler.transform(x_np[start:start + batch_size])
            ).to(device)
            y_log = od_scaler.renormalize(net(xb).squeeze().cpu().numpy())
            preds.append(np.atleast_1d(np.expm1(np.maximum(y_log, 0.0))))
        return np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)

    # Pre-compute validation tensors on GPU (avoid repeated CPU->GPU transfers)
    val_tensors = []
    for xv, yv in zip(xs_valid, ys_valid):
        nz_v = yv > 0
        if not nz_v.any():
            continue
        xv_scaled = feat_scaler.transform(xv[nz_v])
        yv_target = od_scaler.normalize(np.log1p(yv[nz_v]))
        val_tensors.append((
            torch.FloatTensor(xv_scaled).to(device),
            torch.FloatTensor(yv_target).to(device),
        ))

    best_vl = np.inf
    best_pat = patience
    best_state = None
    train_losses = []
    val_losses = []
    val_cpc_vals = []
    val_cpc_fulls = []
    train_cpc_full_hist = []
    train_cpc_nz_hist = []
    val_cpc_full_hist = []
    val_cpc_nz_hist = []
    pbar = tqdm(range(max_epochs), desc='DGM', unit='ep', disable=not verbose)
    for ep in pbar:
        net.train()
        ep_losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = torch.mean((net(xb).squeeze() - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            ep_losses.append(loss.item())

        net.eval()
        with torch.no_grad():
            vls = []
            for xv_t, yv_t in val_tensors:
                yh = net(xv_t).squeeze()
                vls.append(((yh - yv_t) ** 2).mean().item())
            vl = float(np.mean(vls)) if vls else np.inf

            vc_vals = []
            for xv, yv in zip(xs_valid, ys_valid):
                pred_val = _predict_np(xv)
                vc_vals.append(compute_metrics(pred_val, yv)['CPC'])

            vcpcs = []
            full_preds = []
            for xv_full, yv_full in zip(xs_valid_full, ys_valid_full):
                pred_full = _predict_np(xv_full)
                full_preds.append(pred_full)
                vcpcs.append(compute_metrics(pred_full, yv_full)['CPC'])
            vc_val = float(np.mean(vc_vals)) if vc_vals else 0.0
            vc = float(np.mean(vcpcs)) if vcpcs else 0.0

            split_metrics = None
            if single_city_split_ready:
                pred_full = (
                    full_preds[0]
                    if len(full_preds) == 1 and full_preds[0].shape[0] == x_full.shape[0]
                    else _predict_np(x_full)
                )
                split_metrics = masked_train_val_cpc_metrics(
                    pred_full.reshape(od_matrix.shape),
                    od_matrix,
                    train_mask,
                    val_mask,
                    train_full_mask=train_full_mask,
                    val_full_mask=val_full_mask,
                )
            elif multi_city_split_ready:
                train_pred_mats = []
                train_od_mats = []
                for xt, yt in zip(xs_train_full, ys_train_full):
                    nn = int(np.sqrt(yt.shape[0]))
                    train_pred_mats.append(_predict_np(xt).reshape(nn, nn))
                    train_od_mats.append(yt.reshape(nn, nn))
                val_pred_mats = []
                val_od_mats = []
                for xv, yv in zip(xs_valid_full, ys_valid_full):
                    nn = int(np.sqrt(yv.shape[0]))
                    val_pred_mats.append(_predict_np(xv).reshape(nn, nn))
                    val_od_mats.append(yv.reshape(nn, nn))
                split_metrics = {
                    **average_matrix_split_metrics(train_pred_mats, train_od_mats, 'train'),
                    **average_matrix_split_metrics(val_pred_mats, val_od_mats, 'val'),
                }

        tl = float(np.mean(ep_losses))
        train_losses.append(tl)
        val_losses.append(vl)
        val_cpc_vals.append(vc_val)
        val_cpc_fulls.append(vc)
        if split_metrics is not None:
            train_cpc_full_hist.append(split_metrics['CPC_train_full'])
            train_cpc_nz_hist.append(split_metrics['CPC_train_nz'])
            val_cpc_full_hist.append(split_metrics['CPC_val_full'])
            val_cpc_nz_hist.append(split_metrics['CPC_val_nz'])
            postfix = dict(
                loss=f'{tl:.4g}',
                val=f'{vl:.4g}',
                CPC_train_full=f"{split_metrics['CPC_train_full']:.4g}",
                CPC_val_full=f"{split_metrics['CPC_val_full']:.4g}",
                CPC_train_nz=f"{split_metrics['CPC_train_nz']:.4g}",
                CPC_val_nz=f"{split_metrics['CPC_val_nz']:.4g}",
                pat=best_pat,
            )
        else:
            train_cpc_full_hist.append(float('nan'))
            train_cpc_nz_hist.append(float('nan'))
            val_cpc_full_hist.append(float('nan'))
            val_cpc_nz_hist.append(float('nan'))
            postfix = dict(
                loss=f'{tl:.4g}',
                val=f'{vl:.4g}',
                CPC_val=f'{vc_val:.4g}',
                CPC_full=f'{vc:.4g}',
                pat=best_pat,
            )
        pbar.set_postfix(**postfix)

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

    saved_plot_path = save_loss_plot(
        train_losses,
        val_losses,
        title="DGM Training Loss",
        save_path=loss_plot_path,
    )
    if saved_plot_path is not None and verbose:
        print(f"  -> Loss plot saved to {saved_plot_path}")

    predict = _make_predict_fn(net, feat_scaler, od_scaler, device, batch_size)

    predict.train_losses = train_losses
    predict.val_losses = val_losses
    predict.val_cpc_vals = val_cpc_vals
    predict.val_cpc_fulls = val_cpc_fulls
    predict.train_cpc_full_hist = train_cpc_full_hist
    predict.train_cpc_nz_hist = train_cpc_nz_hist
    predict.val_cpc_full_hist = val_cpc_full_hist
    predict.val_cpc_nz_hist = val_cpc_nz_hist
    predict.loss_plot_path = str(saved_plot_path) if saved_plot_path is not None else None

    return predict


def save_model(model, path):
    """Persist a trained DeepGravity predictor."""
    bundle = getattr(model, '_serialization', None)
    if bundle is None:
        raise ValueError("DeepGravity predictor is missing serialization metadata")
    torch.save(bundle, path)


def load_model(path, device=None, **kwargs):
    """Load a persisted DeepGravity predictor."""
    del kwargs
    import os
    import sys

    sys.modules.pop('model', None)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import DeepGravity, OD_normer

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bundle = torch.load(path, map_location='cpu', weights_only=False)
    net = DeepGravity(bundle['input_dim']).to(device)
    net.load_state_dict(bundle['state_dict'])
    feat_scaler = bundle['feat_scaler']
    od_scaler = OD_normer(bundle['od_scaler_min'], bundle['od_scaler_max'])
    return _make_predict_fn(net, feat_scaler, od_scaler, device, bundle['batch_size'])
