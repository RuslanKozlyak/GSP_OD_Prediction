from models.shared.metrics import average_listed_metrics

from .config import BASELINE_MODELS, cleanup_gpu
from .data_utils import split_multi_city_ids
from .gps_loader import GPSBenchmarkLoader
from .runners import run_diffusion_model, run_flat_model, run_graph_model, run_transflower_orig



def _run_baseline_model(model_name, train_areas, valid_areas, test_areas, data_path,
                        gps_loader=None, city_ids=None):
    if model_name in ("RF", "SVR", "GBRT", "DGM", "GM_E", "GM_P"):
        return run_flat_model(model_name, train_areas, valid_areas, test_areas, data_path)
    if model_name in ("GMEL", "GMEL_GBRT", "GMEL_LGBM", "NetGAN"):
        return run_graph_model(model_name, train_areas, valid_areas, test_areas, data_path)
    if model_name in ("DiffODGen", "WeDAN"):
        return run_diffusion_model(model_name, train_areas, valid_areas, test_areas, data_path)
    if model_name == "TransFlowerOrig":
        return run_transflower_orig(
            train_areas, valid_areas, test_areas, data_path,
            gps_loader=gps_loader, city_ids=city_ids,
        )
    raise ValueError(f"Unknown model: {model_name}")



def run_single_city_benchmark(
    gps_run_ids,
    lgbm_run_ids,
    single_city_id,
    data_path,
    baseline_models=None,
    gps_loader=None,
    gmel_gps_run_ids=None,
):
    gps_loader = gps_loader or GPSBenchmarkLoader(single_city_id=single_city_id, data_path=data_path)
    baseline_models = list(BASELINE_MODELS if baseline_models is None else baseline_models)

    results = {}
    model_types = {}

    print("\n[Our Model - GPS variants]")
    for run_id in gps_run_ids:
        metrics = gps_loader.load_gps_results(run_id, area_id=single_city_id)
        if metrics:
            results[run_id] = metrics
            model_types[run_id] = "Ours (GPS)"
    cleanup_gpu()

    print("\n[Our Model - GPS+LGBM variants]")
    for run_id in lgbm_run_ids:
        try:
            metrics = gps_loader.load_lgbm_results(run_id, area_id=single_city_id)
            if metrics:
                results[run_id] = metrics
                model_types[run_id] = "Ours (GPS+LGBM)"
        except Exception as exc:
            print(f"  ERROR {run_id}: {exc}")
        finally:
            cleanup_gpu()

    if gmel_gps_run_ids:
        print("\n[Our Model - GMEL_GPS variants]")
        for run_id in gmel_gps_run_ids:
            metrics = gps_loader.load_gmel_gps_results(run_id, area_id=single_city_id)
            if metrics:
                results[run_id] = metrics
                model_types[run_id] = "Ours (GMEL_GPS)"
        cleanup_gpu()

    print("\n[Baselines - classical & graph models]")
    train_areas = [single_city_id]
    valid_areas = [single_city_id]
    test_areas = [single_city_id]
    for model_name in baseline_models:
        try:
            metrics_list = _run_baseline_model(
                model_name, train_areas, valid_areas, test_areas, data_path,
                gps_loader=gps_loader,
            )
            if metrics_list:
                results[model_name] = average_listed_metrics(metrics_list)
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
):
    gps_loader = gps_loader or GPSBenchmarkLoader(multi_city_ids=city_ids, data_path=data_path)
    baseline_models = list(BASELINE_MODELS if baseline_models is None else baseline_models)
    mc_train, mc_valid, mc_test = split_multi_city_ids(city_ids)

    results = {}
    model_types = {}

    print("\n[Our Model - GPS variants]")
    for run_id in gps_run_ids:
        per_city_metrics = gps_loader.load_multi_city_gps_results(run_id, city_ids=city_ids)
        if per_city_metrics:
            results[run_id] = average_listed_metrics(per_city_metrics)
            model_types[run_id] = "Ours (GPS)"
    cleanup_gpu()

    print("\n[Baselines - classical & graph models]")
    for model_name in baseline_models:
        try:
            metrics_list = _run_baseline_model(
                model_name, mc_train, mc_valid, mc_test, data_path,
                gps_loader=gps_loader, city_ids=city_ids,
            )
            if metrics_list:
                results[model_name] = average_listed_metrics(metrics_list)
                model_types[model_name] = "Baseline"
        except Exception as exc:
            print(f"  ERROR {model_name}: {exc}")
        finally:
            cleanup_gpu()

    return results, model_types, {"train": mc_train, "valid": mc_valid, "test": mc_test}
