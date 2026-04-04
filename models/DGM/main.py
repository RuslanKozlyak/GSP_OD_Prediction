import time
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from models.shared.metrics import compute_metrics


def train(x_train, y_train, xs_valid, ys_valid, xs_valid_full=None, ys_valid_full=None,
          device=None, batch_size=50_000, max_epochs=300, patience=100):
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
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

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
    pbar = tqdm(range(max_epochs), desc='DGM', unit='ep')
    for ep in pbar:
        net.train()
        ep_losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = torch.mean((net(xb).squeeze() - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
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
            for xv_full, yv_full in zip(xs_valid_full, ys_valid_full):
                pred_full = _predict_np(xv_full)
                vcpcs.append(compute_metrics(pred_full, yv_full)['CPC'])
            vc_val = float(np.mean(vc_vals)) if vc_vals else 0.0
            vc = float(np.mean(vcpcs)) if vcpcs else 0.0

        tl = float(np.mean(ep_losses))
        train_losses.append(tl)
        val_losses.append(vl)
        val_cpc_vals.append(vc_val)
        val_cpc_fulls.append(vc)
        pbar.set_postfix(
            loss=f'{tl:.4g}',
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

    # Capture in closure so callers can safely del their local references
    _net, _fs, _os = net, feat_scaler, od_scaler

    def predict(x):
        _net.eval()
        with torch.no_grad():
            y_log = _os.renormalize(
                _net(torch.FloatTensor(_fs.transform(x)).to(device)).squeeze().cpu().numpy()
            )
            return np.atleast_1d(np.expm1(np.maximum(y_log, 0.0)))

    predict.train_losses = train_losses
    predict.val_losses = val_losses
    predict.val_cpc_vals = val_cpc_vals
    predict.val_cpc_fulls = val_cpc_fulls

    return predict


if __name__ == '__main__':
    from pprint import pprint
    import matplotlib.pyplot as plt
    from models.shared.metrics import cal_od_metrics, average_listed_metrics
    from models.shared.data_load import prepare_single_city_flat

    print("\n  **Loading data...")
    data = prepare_single_city_flat()

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict = train(data['x_train'], data['y_train'],
                    data['xs_val'], data['ys_val'],
                    xs_valid_full=data.get('xs_val_full'),
                    ys_valid_full=data.get('ys_val_full'),
                    device=device_)

    # Plot loss curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(predict.train_losses, label='Train Loss')
    ax.plot(predict.val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('DeepGravity Training')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('dgm_loss.png', dpi=150)
    plt.show()
    print("  Loss plot saved to dgm_loss.png")

    print("\n  **Evaluating...")
    metrics_all = []
    for x_one, y_one in zip(data['xs_test'], data['ys_test']):
        n = int(np.sqrt(y_one.shape[0]))
        y_hat = predict(x_one).reshape(n, n)
        y_one = y_one.reshape(n, n)
        y_hat[y_hat < 0] = 0
        metrics_all.append(cal_od_metrics(y_hat, y_one))

    pprint(average_listed_metrics(metrics_all))
