import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from models.shared.metrics import cal_od_metrics, average_listed_metrics


def train(x_train, y_train, xs_valid=None, ys_valid=None, **kwargs):
    """Train GBRT on flat OD pair features.

    Args:
        x_train: (N, F) feature array
        y_train: (N,) target array

    Returns:
        model with .predict(x) method
    """
    model = GradientBoostingRegressor(
        n_estimators=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_depth=None,
    )
    print('  GBRT: fitting...')
    t0 = time.time()
    model.fit(x_train, y_train)
    print(f'  GBRT: fitted in {time.time() - t0:.1f}s')
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


if __name__ == '__main__':
    from pprint import pprint
    from models.shared.data_load import prepare_single_city_flat

    print("\n  **Loading data...")
    data = prepare_single_city_flat()
    model = train(data['x_train'], data['y_train'])

    print("\n  **Evaluating...")
    metrics_all = evaluate(model, data['xs_test'], data['ys_test'])
    pprint(average_listed_metrics(metrics_all))
