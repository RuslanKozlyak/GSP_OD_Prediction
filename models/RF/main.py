import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.shared.metrics import cal_od_metrics, average_listed_metrics


def train(x_train, y_train, xs_valid=None, ys_valid=None, **kwargs):
    """Train Random Forest on flat OD pair features.

    Args:
        x_train: (N, F) feature array
        y_train: (N,) target array

    Returns:
        model with .predict(x) method
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
    )
    print(f'  RF: fitting on {x_train.shape[0]:,} samples...')
    t0 = time.time()
    model.fit(x_train, y_train)
    print(f'  RF: fitted in {time.time() - t0:.1f}s')
    return model


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
