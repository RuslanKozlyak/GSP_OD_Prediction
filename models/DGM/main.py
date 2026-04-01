import time
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm


def train(x_train, y_train, xs_valid, ys_valid,
          device=None, batch_size=50_000, max_epochs=300, patience=100):
    """Train DeepGravity on pre-built feature arrays.

    Args:
        x_train: np.ndarray (N, F) — concatenated pair features for training
        y_train: np.ndarray (N,)   — OD values for training
        xs_valid: list of np.ndarray — per-city validation features
        ys_valid: list of np.ndarray — per-city validation OD values
        device: torch.device (defaults to cuda if available)
        batch_size: DataLoader mini-batch size
        max_epochs: max training epochs
        patience: early-stopping patience

    Returns:
        predict: callable(x: np.ndarray) -> np.ndarray
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import DeepGravity, OD_normer

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_scaler = MinMaxScaler((-1, 1)).fit(x_train)
    od_scaler = OD_normer(y_train.min(), y_train.max())

    ds = TensorDataset(
        torch.FloatTensor(feat_scaler.transform(x_train)),
        torch.FloatTensor(od_scaler.normalize(y_train)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    del ds; gc.collect()

    net = DeepGravity().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    best_vl = np.inf
    best_pat = patience
    pbar = tqdm(range(max_epochs), desc='DGM', unit='ep')
    for ep in pbar:
        net.train()
        ep_losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = torch.mean((net(xb).squeeze() - yb) ** 2)
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        net.eval()
        with torch.no_grad():
            vls = []
            for xv, yv in zip(xs_valid, ys_valid):
                xv_t = torch.FloatTensor(feat_scaler.transform(xv)).to(device)
                yh = net(xv_t).squeeze().cpu().numpy()
                vls.append(((yh - od_scaler.normalize(yv)) ** 2).mean())
            vl = float(np.mean(vls))

        pbar.set_postfix(loss=f'{np.mean(ep_losses):.4g}', val=f'{vl:.4g}', pat=best_pat)

        if vl < best_vl:
            best_vl = vl
            best_pat = patience
        else:
            best_pat -= 1
            if best_pat == 0:
                break

    # Capture in closure so callers can safely del their local references
    _net, _fs, _os = net, feat_scaler, od_scaler

    def predict(x):
        _net.eval()
        with torch.no_grad():
            return _os.renormalize(
                _net(torch.FloatTensor(_fs.transform(x)).to(device)).squeeze().cpu().numpy()
            )

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
        y_hat[y_hat < 0] = 0
        metrics_all.append(cal_od_metrics(y_hat, y_one))

    pprint(average_listed_metrics(metrics_all))
