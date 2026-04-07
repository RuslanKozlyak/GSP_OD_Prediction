import time

import numpy as np
import joblib
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from models.shared.metrics import cal_od_metrics, average_listed_metrics


def train(x_train, y_train, xs_valid=None, ys_valid=None, **kwargs):
    """Train SVR on flat OD pair features.

    Args:
        x_train: (N, F) feature array
        y_train: (N,) target array

    Returns:
        model with .predict(x) method
    """
    n_samples = x_train.shape[0]
    print(f'  SVR: {n_samples:,} samples, fitting LinearSVR...')
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', LinearSVR(C=100, max_iter=10_000)),
    ])
    t0 = time.time()
    model.fit(x_train, y_train)
    print(f'  SVR: fitted in {time.time() - t0:.1f}s')
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
    """Persist a trained SVR model."""
    joblib.dump(model, path)


def load_model(path, **kwargs):
    """Load a persisted SVR model."""
    del kwargs
    return joblib.load(path)
