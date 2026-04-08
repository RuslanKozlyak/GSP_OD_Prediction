from collections import defaultdict
import multiprocessing as mp
from numbers import Real
from pathlib import Path

import numpy as np

from models.shared.metrics import format_train_val_cpc_metrics

from .config import (
    BASELINE_MODELS,
    BASELINE_TRAIN_TIMEOUT_SECONDS,
    INFERENCE_SEEDS,
    SINGLE_CITY_IDS,
    WEIGHTS_DIR,
    cleanup_gpu,
)
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


def _train_baseline_model_worker(result_queue, model_name, train_areas, valid_areas,
                                 test_areas, data_path, city_ids):
    try:
        run_id = _train_baseline_model(
            model_name,
            train_areas,
            valid_areas,
            test_areas,
            data_path,
            gps_loader=None,
            city_ids=city_ids,
        )
        result_queue.put(('ok', run_id))
    except BaseException as exc:
        result_queue.put(('error', repr(exc)))
    finally:
        cleanup_gpu()


def _train_baseline_model_with_timeout(model_name, train_areas, valid_areas, test_areas,
                                       data_path, gps_loader=None, city_ids=None,
                                       timeout_seconds=BASELINE_TRAIN_TIMEOUT_SECONDS):
    if timeout_seconds is None or timeout_seconds <= 0:
        return _train_baseline_model(
            model_name,
            train_areas,
            valid_areas,
            test_areas,
            data_path,
            gps_loader=gps_loader,
            city_ids=city_ids,
        )

    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_train_baseline_model_worker,
        args=(
            result_queue,
            model_name,
            list(train_areas),
            list(valid_areas),
            list(test_areas),
            data_path,
            list(city_ids) if city_ids is not None else None,
        ),
    )
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        print(
            f"  TIMEOUT {model_name}: training exceeded "
            f"{timeout_seconds / 3600:.2f}h; stopping process"
        )
        proc.terminate()
        proc.join(30)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return None

    if result_queue.empty():
        if proc.exitcode == 0:
            return None
        raise RuntimeError(f"{model_name} training process exited with code {proc.exitcode}")

    status, payload = result_queue.get()
    if status == 'error':
        raise RuntimeError(payload)
    return payload


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


def _missing_single_city_gps_weights(base_run_id, single_city_ids, weights_dir):
    weights_dir = Path(WEIGHTS_DIR if weights_dir is None else weights_dir)
    missing = []
    for city_id in single_city_ids:
        run_id = single_city_run_id(base_run_id, city_id)
        weight_path = weights_dir / f"{run_id}.pt"
        if not weight_path.exists():
            missing.append((city_id, weight_path))
    return missing


def _skip_if_single_city_gps_incomplete(base_run_id, single_city_ids, weights_dir):
    missing = _missing_single_city_gps_weights(base_run_id, single_city_ids, weights_dir)
    if not missing:
        return False
    missing_preview = ", ".join(city_id for city_id, _ in missing[:5])
    if len(missing) > 5:
        missing_preview += ", ..."
    print(
        f"  [SKIP] {base_run_id}: missing GPS weights for "
        f"{len(missing)}/{len(single_city_ids)} cities ({missing_preview}). "
        "Train these city-suffixed runs in gps_od.ipynb first."
    )
    print(f"         first missing path: {missing[0][1]}")
    return True


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
    if not baseline_models:
        print("  SKIP: no baseline models configured (BASELINE_MODELS is empty)")
        return trained
    for model_name in baseline_models:
        try:
            if model_name in ("DiffODGen", "WeDAN"):
                print(f"  SKIP {model_name}: training/inference is still coupled for subprocess models")
                continue
            run_ids = []
            for city_id in single_city_ids:
                run_id = _train_baseline_model_with_timeout(
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
    if not baseline_models:
        print("  SKIP: no baseline models configured (BASELINE_MODELS is empty)")
        return trained, {"train": mc_train, "valid": mc_valid, "test": mc_test}
    for model_name in baseline_models:
        try:
            if model_name in ("DiffODGen", "WeDAN"):
                print(f"  SKIP {model_name}: training/inference is still coupled for subprocess models")
                continue
            run_id = _train_baseline_model_with_timeout(
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
        if _skip_if_single_city_gps_incomplete(base_run_id, single_city_ids, gps_weights_dir):
            continue
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
                        metric_groups.get("train"),
                        metric_groups.get("val"),
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
    numeric_keys = sorted({
        key
        for metric in metric_dicts
        for key, value in metric.items()
        if isinstance(value, Real) and not isinstance(value, bool)
    })
    averaged = {}
    for key in numeric_keys:
        values = [
            float(metric[key])
            for metric in metric_dicts
            if key in metric
            and isinstance(metric[key], Real)
            and not isinstance(metric[key], bool)
            and float(metric[key]) == float(metric[key])
        ]
        averaged[key] = float(np.mean(values)) if values else float('nan')
    return averaged


def _average_multi_city_metrics(
    all_metric_dicts,
    test_metric_dicts=None,
    train_metric_dicts=None,
    val_metric_dicts=None,
):
    averaged = _average_metrics(all_metric_dicts)
    if test_metric_dicts:
        averaged_test = _average_metrics(test_metric_dicts)
        for key in ("CPC_test", "MAE_test", "RMSE_test"):
            if key in averaged_test:
                averaged[key] = averaged_test[key]
    _add_split_cpc_metrics(averaged, train_metric_dicts, "train")
    _add_split_cpc_metrics(averaged, val_metric_dicts, "val")
    return averaged


def _add_split_cpc_metrics(target, metric_dicts, prefix):
    if not metric_dicts:
        return
    averaged = _average_metrics(metric_dicts)
    if "CPC_full" in averaged:
        target[f"CPC_{prefix}_full"] = averaged["CPC_full"]
    if "CPC_nz" in averaged:
        target[f"CPC_{prefix}_nz"] = averaged["CPC_nz"]
    if prefix == "val" and "CPC_val_nz" in target:
        target["CPC_val"] = target["CPC_val_nz"]


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
            f"CPC_full={_fmt_metric(city_metrics, 'CPC_full')}  "
            f"CPC_nz={_fmt_metric(city_metrics, 'CPC_nz')}  "
            f"CPC_test={_fmt_metric(city_metrics, 'CPC_test')}  "
            f"MAE_full={_fmt_metric(city_metrics, 'MAE_full')}  "
            f"RMSE_full={_fmt_metric(city_metrics, 'RMSE_full')}"
        )
    print(
        "  Avg all cities: "
        f"CPC_full={_fmt_metric(overall_metrics, 'CPC_full')}  "
        f"CPC_nz={_fmt_metric(overall_metrics, 'CPC_nz')}  "
        f"CPC_test={_fmt_metric(overall_metrics, 'CPC_test')}  "
        f"MAE_full={_fmt_metric(overall_metrics, 'MAE_full')}  "
        f"RMSE_full={_fmt_metric(overall_metrics, 'RMSE_full')}"
    )
    if "CPC_train_full" in overall_metrics:
        print(f"  Train/Val: {format_train_val_cpc_metrics(overall_metrics)}")


def _fmt_metric(metrics, key, precision=4):
    candidates = (key,)
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
