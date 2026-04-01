import time
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def train(x_train, y_train, xs_valid, ys_valid,
          device=None, batch_size=50_000, max_epochs=10000, patience=100):
    """Train GRAVITY (GM_P) on pre-built feature arrays.

    Args:
        x_train: np.ndarray (N, F) — pair features (pop + distance)
        y_train: np.ndarray (N,)   — OD values
        xs_valid: list of np.ndarray — per-city validation features
        ys_valid: list of np.ndarray — per-city validation OD values
        device: torch.device
        batch_size: DataLoader mini-batch size
        max_epochs / patience: training schedule

    Returns:
        predict: callable(x: np.ndarray) -> np.ndarray (abs applied)
    """
    import os, sys
    sys.modules.pop('model', None)  # prevent collision when run after another model
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import GRAVITY

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalise distance (col 2) to [0,1] — raw metres cause exp(-69300)≈0 at gamma=0.5
    x_train = x_train.copy()
    dist_max = float(x_train[:, 2].max())
    if dist_max > 1.0:
        x_train[:, 2] /= dist_max
    xs_valid = [xv.copy() for xv in xs_valid]
    for xv in xs_valid:
        if dist_max > 1.0:
            xv[:, 2] /= dist_max

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
    # xs_valid is already distance-normalized at this point
    _vx = [xv[yv > 0] for xv, yv in zip(xs_valid, ys_valid) if (yv > 0).any()]
    _vy = [np.log1p(yv[yv > 0]) for xv, yv in zip(xs_valid, ys_valid) if (yv > 0).any()]
    if _vx:
        xv_all_t = torch.FloatTensor(np.concatenate(_vx)).to(device)
        yv_log_all = np.concatenate(_vy)
    else:
        xv_all_t = None
    del _vx, _vy

    net = GRAVITY().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    best_vl = np.inf
    best_pat = patience
    pbar = tqdm(range(max_epochs), desc='GM_P', unit='ep')
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
                yh_all = net(xv_all_t).squeeze().cpu().numpy()
                vl = float(((np.log1p(yh_all) - yv_log_all) ** 2).mean())
            else:
                vl = np.inf

        pbar.set_postfix(loss=f'{np.mean(ep_losses):.4g}', val=f'{vl:.4g}', pat=best_pat)

        if vl < best_vl:
            best_vl = vl
            best_pat = patience
        else:
            best_pat -= 1
            if best_pat == 0:
                break

    _net, _dist_max = net, dist_max

    def predict(x):
        _net.eval()
        x = x.copy()
        if _dist_max > 1.0:
            x[:, 2] /= _dist_max
        with torch.no_grad():
            return np.abs(_net(torch.FloatTensor(x).to(device)).squeeze().cpu().numpy())

    return predict


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_load import load_data
    from metrics import cal_od_metrics, average_listed_metrics
    from pprint import pprint

    print("\n  **Loading data...")
    xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data()

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict = train(xtrain, ytrain, xvalid, yvalid, device=device_)

    print("\n  **Evaluating...")
    metrics_all = []
    for x_one, y_one in zip(xtest, ytest):
        n = int(np.sqrt(y_one.shape[0]))
        y_hat = predict(x_one).reshape(n, n)
        y_one = y_one.reshape(n, n)
        metrics_all.append(cal_od_metrics(y_hat, y_one))

    pprint(average_listed_metrics(metrics_all))
