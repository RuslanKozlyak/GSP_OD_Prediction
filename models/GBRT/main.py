import time

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from tqdm.auto import tqdm

from models.shared.metrics import cal_od_metrics, average_listed_metrics


def train(x_train, y_train, xs_valid=None, ys_valid=None, **kwargs):
    """Train GBRT on flat OD pair features.

    Uses the ``monitor`` callback of ``GradientBoostingRegressor.fit`` to drive
    a tqdm bar that updates after each boosting round — so you can see how
    many trees remain and the running training loss.

    Args:
        x_train: (N, F) feature array
        y_train: (N,) target array

    Returns:
        model with .predict(x) method
    """
    verbose = int(kwargs.get('verbose', 1) or 0)
    n_estimators = int(kwargs.get('n_estimators', 20))
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        min_samples_split=kwargs.get('min_samples_split', 2),
        min_samples_leaf=kwargs.get('min_samples_leaf', 2),
        max_depth=None,
        verbose=0,  # replaced by our tqdm monitor
        random_state=kwargs.get('random_state', 42),
    )
    if verbose:
        print(f'  GBRT: fitting {n_estimators} trees on {x_train.shape[0]:,} samples x {x_train.shape[1]} features')

    pbar = tqdm(total=n_estimators, desc='GBRT', unit='tree', disable=not verbose)

    def _monitor(i, est, _locals):
        # ``i`` is 0-based; update bar to (i+1)/n_estimators and show train loss.
        pbar.update(1)
        try:
            train_loss = float(est.train_score_[i])
            pbar.set_postfix(train_loss=f'{train_loss:.4g}')
        except Exception:
            pass
        return False  # do not early-stop

    t0 = time.time()
    try:
        model.fit(x_train, y_train, monitor=_monitor)
    finally:
        pbar.close()
    if verbose:
        print(f'  GBRT: fitted {n_estimators} trees in {time.time() - t0:.1f}s')
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
    """Persist a trained GBRT model."""
    joblib.dump(model, path)


def load_model(path, **kwargs):
    """Load a persisted GBRT model."""
    del kwargs
    return joblib.load(path)
