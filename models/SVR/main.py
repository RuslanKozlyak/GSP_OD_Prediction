import time

import numpy as np
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler

from models.shared.metrics import cal_od_metrics, average_listed_metrics


class _ScaledSVR:
    """LinearSVR with built-in feature scaling and batched prediction."""

    def __init__(self, model, scaler):
        self._model = model
        self._scaler = scaler

    def predict(self, x):
        CHUNK = 200_000
        parts = []
        for i in range(0, x.shape[0], CHUNK):
            parts.append(self._model.predict(self._scaler.transform(x[i:i + CHUNK])))
        return np.concatenate(parts)


def train(x_train, y_train, xs_valid=None, ys_valid=None, **kwargs):
    """Train LinearSVR on flat OD pair features.

    LinearSVR is O(n) per iteration — feasible for 500k+ samples.
    sklearn's RBF SVR is O(n²-n³) and infeasible at this scale.

    Args:
        x_train: (N, F) feature array
        y_train: (N,) target array

    Returns:
        model with .predict(x) method
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = LinearSVR(C=1.0, max_iter=5000, tol=1e-4)
    print(f'  SVR: fitting LinearSVR on {x_train.shape[0]:,} samples...')
    t0 = time.time()
    model.fit(x_scaled, y_train)
    print(f'  SVR: fitted in {time.time() - t0:.1f}s')
    return _ScaledSVR(model, scaler)


def evaluate(model, xs_test, ys_test):
    """Evaluate on test data, return list of per-area metric dicts."""
    metrics_all = []
    for x_one, y_one in zip(xs_test, ys_test):
        n = int(np.sqrt(x_one.shape[0]))
        y_hat = model.predict(x_one).reshape(n, n)
        y_true = y_one.reshape(n, n)
        y_hat[y_hat < 0] = 0
        metrics_all.append(cal_od_metrics(y_hat, y_true))
    return metrics_all
