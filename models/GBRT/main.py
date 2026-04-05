import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from models.shared.metrics import cal_od_metrics, average_listed_metrics


def train(x_train, y_train, xs_valid=None, ys_valid=None, **kwargs):
    """Train GBRT on flat OD pair features.

    Uses HistGradientBoostingRegressor which is O(n) per split (bins features)
    and handles 500k+ samples efficiently.

    Args:
        x_train: (N, F) feature array
        y_train: (N,) target array

    Returns:
        model with .predict(x) method
    """
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        max_leaf_nodes=63,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    print(f'  GBRT: fitting HistGBRT on {x_train.shape[0]:,} samples...')
    t0 = time.time()
    model.fit(x_train, y_train)
    print(f'  GBRT: fitted in {time.time() - t0:.1f}s  '
          f'(n_iter={model.n_iter_} best_iter={getattr(model, "n_iter_", "?")})')
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
