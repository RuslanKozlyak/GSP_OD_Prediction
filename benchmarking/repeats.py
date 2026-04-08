from numbers import Real

import numpy as np


def single_city_run_id(base_run_id, city_id):
    """Build a per-city artifact id for single-city experiments."""
    return f"{base_run_id}__city_{city_id}"


def single_city_lgbm_run_id(base_run_id, city_id):
    """Build a per-city LGBM artifact id from the donor base id."""
    base = base_run_id[:-5] if base_run_id.endswith("_lgbm") else base_run_id
    return f"{single_city_run_id(base, city_id)}_lgbm"


def aggregate_metric_samples(metric_samples):
    """Aggregate repeated metric dicts into mean/std columns."""
    if not metric_samples:
        return {}

    numeric_keys = sorted({
        key
        for sample in metric_samples
        for key, value in sample.items()
        if _is_numeric(value)
    })

    aggregated = {"n_runs": len(metric_samples)}
    for key in numeric_keys:
        values = [
            float(sample[key])
            for sample in metric_samples
            if key in sample and _is_numeric(sample[key]) and float(sample[key]) == float(sample[key])
        ]
        if not values:
            aggregated[key] = float('nan')
            aggregated[f"{key}_std"] = float('nan')
            continue
        arr = np.asarray(values, dtype=float)
        aggregated[key] = float(np.mean(arr))
        aggregated[f"{key}_std"] = float(np.std(arr))

    return aggregated


def _is_numeric(value):
    return isinstance(value, Real) and not isinstance(value, bool)
