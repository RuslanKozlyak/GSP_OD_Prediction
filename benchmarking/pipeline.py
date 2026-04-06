from collections import defaultdict
from numbers import Real

from .config import BASELINE_MODELS, INFERENCE_SEEDS, SINGLE_CITY_IDS, WEIGHTS_DIR, cleanup_gpu
from .data_utils import split_multi_city_ids
from .gps_loader import GPSBenchmarkLoader
from .repeats import aggregate_metric_samples, single_city_lgbm_run_id, single_city_run_id
from .runners import (
    infer_flat_model,
    infer_graph_model,
    infer_transflower_orig,
    run_diffusion_model,
    train_flat_model,
    train_graph_model,
    train_transflower_orig,
)



def _train_baseline_model(model_name, train_areas, valid_areas, test_areas, data_path,
                          gps_loader=None, city_ids=None):
    if model_name in ("RF", "SVR", "GBRT", "DGM", "GM_E", "GM_P"):
        return train_flat_model(
            model_name, train_areas, valid_areas, test_areas, data_path,
        )
    if model_name in ("GMEL", "GMEL_GBRT", "GMEL_LGBM", "NetGAN"):
        return train_graph_model(
            model_name, train_areas, valid_areas, test_areas, data_path,
        )
    if model_name in ("DiffODGen", "WeDAN"):
        return None
    if model_name == "TransFlowerOrig":
        return train_transflower_orig(
            train_areas, valid_areas, test_areas, data_path,
            gps_loader=gps_loader, city_ids=city_ids,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _infer_baseline_model(model_name, train_areas, valid_areas, test_areas, data_path,
                          gps_loader=None, city_ids=None, inference_seeds=None):
    if model_name in ("RF", "SVR", "GBRT", "DGM", "GM_E", "GM_P"):
        return infer_flat_model(
            model_name, train_areas, valid_areas, test_areas, data_path,
            inference_seeds=inference_seeds,
        )
    if model_name in ("GMEL", "GMEL_GBRT", "GMEL_LGBM", "NetGAN"):
        return infer_graph_model(
            model_name, train_areas, valid_areas, test_areas, data_path,
            inference_seeds=inference_seeds,
        )
    if model_name in ("DiffODGen", "WeDAN"):
        return run_diffusion_model(
            model_name, train_areas, valid_areas, test_areas, data_path,
            inference_seeds=inference_seeds,
        )
    if model_name == "TransFlowerOrig":
        return infer_transflower_orig(
            train_areas, valid_areas, test_areas, data_path,
            gps_loader=gps_loader, city_ids=city_ids, inference_seeds=inference_seeds,
        )
    raise ValueError(f"Unknown model: {model_name}")


def train_single_city_benchmark_models(
    single_city_ids,
    data_path,
    baseline_models=None,
    gps_loader=None,
):
    single_city_ids = _normalize_single_city_ids(single_city_ids)
    gps_loader = gps_loader or GPSBenchmarkLoader(single_city_id=single_city_ids[0], data_path=data_path)
    baseline_models = list(BASELINE_MODELS if baseline_models is None else baseline_models)

    trained = {}
    print("\n[Baseline training - single city]")
    for model_name in baseline_models:
        try:
            if model_name in ("DiffODGen", "WeDAN"):
                print(f"  SKIP {model_name}: training/inference is still coupled for subprocess models")
                continue
            run_ids = []
            for city_id in single_city_ids:
                run_id = _train_baseline_model(
                    model_name,
                    [city_id],
                    [city_id],
                    [city_id],
                    data_path,
                    gps_loader=gps_loader,
                )
                if run_id is not None:
                    run_ids.append(run_id)
            if run_ids:
                trained[model_name] = run_ids
        except Exception as exc:
            print(f"  ERROR {model_name}: {exc}")
        finally:
            cleanup_gpu()
    return trained


def train_multi_city_benchmark_models(
    city_ids,
    data_path,
    baseline_models=None,
    gps_loader=None,
):
    gps_loader = gps_loader or GPSBenchmarkLoader(multi_city_ids=city_ids, data_path=data_path)
    baseline_models = list(BASELINE_MODELS if baseline_models is None else baseline_models)
    mc_train, mc_valid, mc_test = split_multi_city_ids(city_ids)

    trained = {}
    print("\n[Baseline training - multi city]")
    for model_name in baseline_models:
        try:
            if model_name in ("DiffODGen", "WeDAN"):
                print(f"  SKIP {model_name}: training/inference is still coupled for subprocess models")
                continue
            run_id = _train_baseline_model(
                model_name,
                mc_train,
                mc_valid,
                mc_test,
                data_path,
                gps_loader=gps_loader,
                city_ids=city_ids,
            )
            if run_id is not None:
                trained[model_name] = run_id
        except Exception as exc:
            print(f"  ERROR {model_name}: {exc}")
        finally:
            cleanup_gpu()
    return trained, {"train": mc_train, "valid": mc_valid, "test": mc_test}



def run_single_city_benchmark(
    gps_run_ids,
    lgbm_run_ids,
    single_city_ids,
    data_path,
    baseline_models=None,
    gps_loader=None,
    gmel_gps_run_ids=None,
    inference_seeds=None,
    gps_weights_dir=WEIGHTS_DIR,
    gps_model_type_label="Ours (GPS)",
):
    single_city_ids = _normalize_single_city_ids(single_city_ids)
    inference_seeds = list(INFERENCE_SEEDS if inference_seeds is None else inference_seeds)
    gps_loader = gps_loader or GPSBenchmarkLoader(single_city_id=single_city_ids[0], data_path=data_path)
    baseline_models = list(BASELINE_MODELS if baseline_models is None else baseline_models)

    results = {}
    model_types = {}

    print("\n[Our Model - GPS variants]")
    for base_run_id in gps_run_ids:
        metric_samples = []
        for city_id in single_city_ids:
            run_id = single_city_run_id(base_run_id, city_id)
            for inference_seed in inference_seeds:
                metrics = gps_loader.load_gps_results(
                    run_id,
                    area_id=city_id,
                    inference_seed=inference_seed,
                    weights_dir=gps_weights_dir,
                )
                if metrics:
                    metric_samples.append(metrics)
        if metric_samples:
            results[base_run_id] = aggregate_metric_samples(metric_samples)
            model_types[base_run_id] = gps_model_type_label
    cleanup_gpu()

    print("\n[Our Model - GPS+LGBM variants]")
    for base_run_id in lgbm_run_ids:
        result_key = f"{base_run_id[:-5] if base_run_id.endswith('_lgbm') else base_run_id}_lgbm"
        metric_samples = []
        try:
            for city_id in single_city_ids:
                run_id = single_city_lgbm_run_id(base_run_id, city_id)
                for inference_seed in inference_seeds:
                    metrics = gps_loader.load_lgbm_results(
                        run_id,
                        area_id=city_id,
                        inference_seed=inference_seed,
                    )
                    if metrics:
                        metric_samples.append(metrics)
            if metric_samples:
                results[result_key] = aggregate_metric_samples(metric_samples)
                model_types[result_key] = "Ours (GPS+LGBM)"
        except Exception as exc:
            print(f"  ERROR {base_run_id}: {exc}")
        finally:
            cleanup_gpu()

    if gmel_gps_run_ids:
        print("\n[Our Model - GMEL_GPS variants]")
        for base_run_id in gmel_gps_run_ids:
            metric_samples = []
            for city_id in single_city_ids:
                run_id = single_city_run_id(base_run_id, city_id)
                for inference_seed in inference_seeds:
                    metrics = gps_loader.load_gmel_gps_results(
                        run_id,
                        area_id=city_id,
                        inference_seed=inference_seed,
                    )
                    if metrics:
                        metric_samples.append(metrics)
            if metric_samples:
                results[base_run_id] = aggregate_metric_samples(metric_samples)
                model_types[base_run_id] = "Ours (GMEL_GPS)"
        cleanup_gpu()

    print("\n[Baselines - classical & graph models]")
    for model_name in baseline_models:
        try:
            metric_samples = []
            for city_id in single_city_ids:
                metrics_list = _infer_baseline_model(
                    model_name,
                    [city_id],
                    [city_id],
                    [city_id],
                    data_path,
                    gps_loader=gps_loader,
                    inference_seeds=inference_seeds,
                )
                if metrics_list:
                    metric_samples.extend(metrics_list)
            if metric_samples:
                results[model_name] = aggregate_metric_samples(metric_samples)
                model_types[model_name] = "Baseline"
        except Exception as exc:
            print(f"  ERROR {model_name}: {exc}")
        finally:
            cleanup_gpu()

    return results, model_types



def run_multi_city_benchmark(
    gps_run_ids,
    city_ids,
    data_path,
    baseline_models=None,
    gps_loader=None,
    inference_seeds=None,
    gps_weights_dir=WEIGHTS_DIR,
    gps_model_type_label="Ours (GPS)",
):
    inference_seeds = list(INFERENCE_SEEDS if inference_seeds is None else inference_seeds)
    gps_loader = gps_loader or GPSBenchmarkLoader(multi_city_ids=city_ids, data_path=data_path)
    baseline_models = list(BASELINE_MODELS if baseline_models is None else baseline_models)
    mc_train, mc_valid, mc_test = split_multi_city_ids(city_ids)

    results = {}
    model_types = {}

    print("\n[Our Model - GPS variants]")
    for run_id in gps_run_ids:
        metric_samples = []
        per_city_metric_samples = defaultdict(list)
        for inference_seed in inference_seeds:
            metric_groups = gps_loader.load_multi_city_gps_results(
                run_id,
                city_ids=city_ids,
                inference_seed=inference_seed,
                evaluate_all_cities=True,
                return_split_groups=True,
                verbose=False,
                weights_dir=gps_weights_dir,
            )
            if metric_groups and metric_groups.get("all"):
                for city_metric in metric_groups["all"]:
                    city_id = city_metric.get("city_id")
                    if city_id is not None:
                        per_city_metric_samples[city_id].append(city_metric)
                metric_samples.append(
                    _average_multi_city_metrics(
                        metric_groups["all"],
                        metric_groups.get("test"),
                    )
                )
        if metric_samples:
            per_city_summary = _summarize_multi_city_per_city(per_city_metric_samples)
            results[run_id] = aggregate_metric_samples(metric_samples)
            results[run_id]["per_city"] = per_city_summary
            model_types[run_id] = gps_model_type_label
            _print_multi_city_city_summary(run_id, per_city_summary, results[run_id])
    cleanup_gpu()

    print("\n[Baselines - classical & graph models]")
    for model_name in baseline_models:
        try:
            metrics_list = _infer_baseline_model(
                model_name,
                mc_train, mc_valid, mc_test, data_path,
                gps_loader=gps_loader, city_ids=city_ids, inference_seeds=inference_seeds,
            )
            if metrics_list:
                results[model_name] = aggregate_metric_samples(metrics_list)
                model_types[model_name] = "Baseline"
        except Exception as exc:
            print(f"  ERROR {model_name}: {exc}")
        finally:
            cleanup_gpu()

    return results, model_types, {"train": mc_train, "valid": mc_valid, "test": mc_test}


def _normalize_single_city_ids(single_city_ids):
    if single_city_ids is None:
        return list(SINGLE_CITY_IDS)
    if isinstance(single_city_ids, str):
        return [single_city_ids]
    return list(single_city_ids)


def _average_metrics(metric_dicts):
    numeric_keys = [
        key for key, value in metric_dicts[0].items()
        if isinstance(value, Real) and not isinstance(value, bool)
    ]
    return {
        key: sum(metric[key] for metric in metric_dicts) / len(metric_dicts)
        for key in numeric_keys
    }


def _average_multi_city_metrics(all_metric_dicts, test_metric_dicts=None):
    averaged = _average_metrics(all_metric_dicts)
    if test_metric_dicts:
        averaged_test = _average_metrics(test_metric_dicts)
        for key in ("CPC_test", "MAE_test", "RMSE_test"):
            if key in averaged_test:
                averaged[key] = averaged_test[key]
    return averaged


def _summarize_multi_city_per_city(per_city_metric_samples):
    summary = {}
    for city_id in sorted(per_city_metric_samples):
        aggregated = aggregate_metric_samples(per_city_metric_samples[city_id])
        aggregated["is_test_city"] = any(
            sample.get("is_test_city", False)
            for sample in per_city_metric_samples[city_id]
        )
        summary[city_id] = aggregated
    return summary


def _print_multi_city_city_summary(run_id, per_city_summary, overall_metrics):
    print(f"  {run_id}:")
    for city_id in sorted(per_city_summary):
        city_metrics = per_city_summary[city_id]
        tag = " [test]" if city_metrics.get("is_test_city") else ""
        print(
            f"    {city_id}{tag}: "
            f"CPC_full={_fmt_metric(city_metrics, 'CPC')}  "
            f"CPC_nonzero={_fmt_metric(city_metrics, 'CPC_nonzero')}  "
            f"CPC_test={_fmt_metric(city_metrics, 'CPC_test')}  "
            f"MAE={_fmt_metric(city_metrics, 'MAE')}  "
            f"RMSE={_fmt_metric(city_metrics, 'RMSE')}"
        )
    print(
        "  Avg all cities: "
        f"CPC_full={_fmt_metric(overall_metrics, 'CPC')}  "
        f"CPC_nonzero={_fmt_metric(overall_metrics, 'CPC_nonzero')}  "
        f"CPC_test={_fmt_metric(overall_metrics, 'CPC_test')}  "
        f"MAE={_fmt_metric(overall_metrics, 'MAE')}  "
        f"RMSE={_fmt_metric(overall_metrics, 'RMSE')}"
    )


def _fmt_metric(metrics, key, precision=4):
    aliases = {
        "CPC_nonzero": ("CPC_nonzero", "CPC_nz"),
    }
    candidates = aliases.get(key, (key,))
    mean = None
    std = None
    for candidate in candidates:
        if candidate in metrics:
            mean = metrics.get(candidate)
            std = metrics.get(f"{candidate}_std")
            break
    if mean is None:
        return "-"
    try:
        mean = float(mean)
    except (TypeError, ValueError):
        return "-"
    if mean != mean:
        return "-"
    if std is None:
        return f"{mean:.{precision}f}"
    try:
        std = float(std)
    except (TypeError, ValueError):
        return f"{mean:.{precision}f}"
    if std != std:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f}+/-{std:.{precision}f}"
