import time

import numpy as np
import joblib
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
    verbose = int(kwargs.get('verbose', 1) or 0)
    model = RandomForestRegressor(
        n_estimators=kwargs.get('n_estimators', 20),
        oob_score=True,
        max_depth=None,
        min_samples_split=kwargs.get('min_samples_split', 2),
        min_samples_leaf=kwargs.get('min_samples_leaf', 2),
        n_jobs=-1,
        verbose=verbose,
    )
    if verbose:
        print('  RF: fitting...')
    t0 = time.time()
    model.fit(x_train, y_train)
    if verbose:
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


def save_model(model, path):
    """Persist a trained Random Forest model."""
    joblib.dump(model, path)


def load_model(path, **kwargs):
    """Load a persisted Random Forest model."""
    del kwargs
    return joblib.load(path)
