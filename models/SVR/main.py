import time

import numpy as np
import joblib
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from models.shared.metrics import cal_od_metrics, average_listed_metrics


def train(x_train, y_train, xs_valid=None, ys_valid=None, **kwargs):
    """Train SVR on flat OD pair features.

    LinearSVR (liblinear backend) does not expose a per-iteration callback, so
    we can't drive a tqdm bar. Instead we pass ``verbose=1`` through to
    liblinear (which prints its optimizer progress to stderr) and report the
    actual number of iterations used vs. ``max_iter`` once fit finishes.

    Args:
        x_train: (N, F) feature array
        y_train: (N,) target array

    Returns:
        model with .predict(x) method
    """
    n_samples, n_features = x_train.shape
    verbose = int(kwargs.get('verbose', 1) or 0)
    max_iter = int(kwargs.get('max_iter', 10_000))
    if verbose:
        print(
            f'  SVR: fitting LinearSVR on {n_samples:,} samples x {n_features} '
            f'features (max_iter={max_iter:,})'
        )
    svr = LinearSVR(
        C=kwargs.get('C', 100),
        loss=kwargs.get('loss', 'squared_epsilon_insensitive'),
        max_iter=max_iter,
        dual=kwargs.get('dual', False),
        random_state=kwargs.get('random_state', 42),
        verbose=verbose,
    )
    model = Pipeline([
        ('scaler', StandardScaler(copy=False)),
        ('svr', svr),
    ])
    t0 = time.time()
    model.fit(x_train, y_train)
    elapsed = time.time() - t0
    if verbose:
        n_iter = getattr(svr, 'n_iter_', None)
        iter_info = (
            f' in {n_iter}/{max_iter} iterations'
            if n_iter is not None else ''
        )
        converged = (
            ' (converged)' if n_iter is not None and n_iter < max_iter
            else ' (hit max_iter — consider raising it)' if n_iter == max_iter
            else ''
        )
        print(f'  SVR: fitted{iter_info} in {elapsed:.1f}s{converged}')
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
