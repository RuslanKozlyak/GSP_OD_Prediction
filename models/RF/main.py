import time

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

from models.shared.metrics import cal_od_metrics, average_listed_metrics


def train(x_train, y_train, xs_valid=None, ys_valid=None, **kwargs):
    """Train Random Forest on flat OD pair features.

    Uses ``warm_start=True`` so we can add trees one by one and drive a tqdm
    bar showing ``tree i/N`` + ETA. With ``n_jobs=-1`` each individual ``fit``
    call still parallelises internally within the single tree being added.

    Args:
        x_train: (N, F) feature array
        y_train: (N,) target array

    Returns:
        model with .predict(x) method
    """
    verbose = int(kwargs.get('verbose', 1) or 0)
    n_estimators = int(kwargs.get('n_estimators', 20))
    model = RandomForestRegressor(
        n_estimators=1,
        oob_score=False,  # oob requires all trees at once; skip for warm_start
        max_depth=None,
        min_samples_split=kwargs.get('min_samples_split', 2),
        min_samples_leaf=kwargs.get('min_samples_leaf', 2),
        n_jobs=-1,
        warm_start=True,
        verbose=0,
        random_state=kwargs.get('random_state', 42),
    )
    if verbose:
        print(f'  RF: fitting {n_estimators} trees on {x_train.shape[0]:,} samples x {x_train.shape[1]} features')
    t0 = time.time()
    pbar = tqdm(range(1, n_estimators + 1), desc='RF', unit='tree', disable=not verbose)
    for i in pbar:
        model.n_estimators = i
        model.fit(x_train, y_train)
        elapsed = time.time() - t0
        pbar.set_postfix(elapsed=f'{elapsed:.1f}s')
    if verbose:
        print(f'  RF: fitted {n_estimators} trees in {time.time() - t0:.1f}s')
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
