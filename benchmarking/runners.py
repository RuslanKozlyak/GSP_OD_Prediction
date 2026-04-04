"""Thin dispatcher that calls models/<name>/main.py for training and evaluation."""
import gc
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from models.shared.metrics import average_listed_metrics, cal_od_metrics
from models.shared.data_load import (
    construct_flat_features, load_graph_data, get_scalers, build_dgl_graph, build_pyg_graph,
    prepare_single_city_flat, prepare_single_city_graph,
)

from .config import DATA_PATH, PROJECT_ROOT, SEED, cleanup_gpu, device


def load_model_main(model_name):
    """Dynamically import a model's main.py module."""
    model_dir = PROJECT_ROOT / "models" / model_name
    spec = importlib.util.spec_from_file_location(f"{model_name}_main", model_dir / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_flat_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    """Train and evaluate a flat-feature model (RF, SVR, GBRT, DGM, GM_E, GM_P)."""
    print(f"\n{'=' * 60}\n  Running: {model_name}\n{'=' * 60}")
    t0 = time.time()
    feature_mode = "gravity" if model_name in ("GM_E", "GM_P") else "full"
    single_city_split = (
        len(train_areas) == len(valid_areas) == len(test_areas) == 1
        and train_areas[0] == valid_areas[0] == test_areas[0]
    )

    if single_city_split:
        split_data = prepare_single_city_flat(
            area_id=train_areas[0], data_path=data_path, feature_mode=feature_mode
        )
        x_train = split_data['x_train']
        y_train = split_data['y_train']
        xs_valid = split_data['xs_val']
        ys_valid = split_data['ys_val']
        xs_valid_full = split_data['xs_val_full']
        ys_valid_full = split_data['ys_val_full']
        xs_test = split_data['xs_test']
        ys_test = split_data['ys_test']
    else:
        xs_tr, ys_tr = construct_flat_features(train_areas, data_path, feature_mode)
        x_train = np.concatenate(xs_tr, axis=0)
        y_train = np.concatenate(ys_tr, axis=0)
        del xs_tr, ys_tr; gc.collect()

        xs_valid, ys_valid = construct_flat_features(valid_areas, data_path, feature_mode)
        xs_valid_full, ys_valid_full = xs_valid, ys_valid
        xs_test, ys_test = construct_flat_features(test_areas, data_path, feature_mode)

    # Train
    module = load_model_main(model_name)
    if hasattr(module, 'train'):
        model = module.train(
            x_train, y_train, xs_valid, ys_valid,
            xs_valid_full=xs_valid_full, ys_valid_full=ys_valid_full,
            device=device, batch_size=10_000, max_epochs=10000, patience=100,
        )
    else:
        raise ValueError(f"Model {model_name} has no train() function")

    del x_train, y_train, xs_valid, ys_valid; gc.collect()

    # Evaluate
    if hasattr(module, 'evaluate'):
        metrics_all = module.evaluate(model, xs_test, ys_test)
    else:
        metrics_all = []
        for x_one, y_one in zip(xs_test, ys_test):
            n_nodes = int(np.sqrt(y_one.shape[0]))
            y_hat = model.predict(x_one) if hasattr(model, 'predict') else model(x_one)
            y_hat = y_hat.reshape(n_nodes, n_nodes)
            y_true = y_one.reshape(n_nodes, n_nodes)
            y_hat[y_hat < 0] = 0
            metrics_all.append(cal_od_metrics(y_hat, y_true))

    avg = average_listed_metrics(metrics_all)
    print(f"  CPC={avg['CPC']:.4f}  RMSE={avg['RMSE']:.2f}  MAE={avg['MAE']:.2f}  ({time.time() - t0:.1f}s)")
    del model; gc.collect(); cleanup_gpu()
    return metrics_all


def run_graph_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    """Train and evaluate a graph-based model (GMEL, NetGAN)."""
    print(f"\n{'=' * 60}\n  Running: {model_name}\n{'=' * 60}")
    t0 = time.time()
    single_city_split = (
        len(train_areas) == len(valid_areas) == len(test_areas) == 1
        and train_areas[0] == valid_areas[0] == test_areas[0]
    )

    if single_city_split:
        nfeat_scaler = dis_scaler = od_scaler = None
    else:
        nf_train, _, dis_train, od_train = load_graph_data(train_areas, data_path)
        nfeat_scaler, dis_scaler, od_scaler = get_scalers(nf_train, dis_train, od_train)
    metrics_all = []

    try:
        if model_name == "GMEL":
            module = load_model_main("GMEL")
            single_city_data = None
            if single_city_split:
                single_city_data = prepare_single_city_graph(train_areas[0], data_path=data_path)
            gmel, gbrt, nfeat_scaler, dis_scaler = module.train(
                train_areas, valid_areas, str(data_path),
                device=device,
                nfeat_scaler=nfeat_scaler, dis_scaler=dis_scaler, od_scaler=od_scaler,
                single_city_data=single_city_data,
                max_epochs=1000, patience=10,
            )
            nf_test, adj_test, dis_test, od_test = load_graph_data(test_areas, data_path)
            gmel.eval()
            for nf, adj, dis, od in tqdm(zip(nf_test, adj_test, dis_test, od_test),
                                          total=len(nf_test), desc="GMEL eval"):
                nf_t = torch.FloatTensor(nfeat_scaler.transform(nf)).to(device)
                graph = build_pyg_graph(adj, device)
                with torch.no_grad():
                    _, _, _, h_in, h_out = gmel(graph, nf_t)
                    h = np.concatenate([h_in.cpu().numpy(), h_out.cpu().numpy()], axis=1)
                    n = h.shape[0]
                    h_o = h.reshape(n, 1, h.shape[1]).repeat(n, axis=1)
                    h_d = h.reshape(1, n, h.shape[1]).repeat(n, axis=0)
                    feat = np.concatenate([h_o, h_d, dis.reshape(n, n, 1)], axis=2)
                    od_hat = gbrt.predict(feat.reshape(-1, h.shape[1] * 2 + 1)).reshape(n, n)
                    od_hat[od_hat < 0] = 0
                metrics_all.append(cal_od_metrics(od_hat, od))
            del gmel, gbrt; gc.collect()

        elif model_name == "NetGAN":
            module = load_model_main("NetGAN")
            trained = module.train(
                train_areas, valid_areas, str(data_path),
                device=device,
                nfeat_scaler=nfeat_scaler, dis_scaler=dis_scaler, od_scaler=od_scaler,
            )
            metrics_all = module.evaluate(trained, test_areas, str(data_path), device=device)
            del trained; gc.collect()

        else:
            raise ValueError(f"Unsupported graph model: {model_name}")
    finally:
        cleanup_gpu()

    avg = average_listed_metrics(metrics_all)
    print(f"  CPC={avg['CPC']:.4f}  RMSE={avg['RMSE']:.2f}  MAE={avg['MAE']:.2f}  ({time.time() - t0:.1f}s)")
    return metrics_all


def run_diffusion_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    """Run DiffODGen/WeDAN as subprocess (complex dependencies)."""
    del train_areas, valid_areas, test_areas, data_path
    print(f"\n{'=' * 60}\n  Running: {model_name}\n{'=' * 60}")
    t0 = time.time()
    model_dir = PROJECT_ROOT / "models" / model_name
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True, timeout=7200, cwd=model_dir,
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-300:]}")
            return []
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 2h")
        return []

    try:
        lines = result.stdout.strip().splitlines()
        dict_lines = []
        in_dict = False
        for line in lines:
            if line.strip().startswith("{"):
                in_dict = True
            if in_dict:
                dict_lines.append(line)
            if in_dict and line.strip().endswith("}"):
                break
        if dict_lines:
            avg_metrics = json.loads("\n".join(dict_lines).replace("'", '"'))
            print(f"  CPC={avg_metrics.get('CPC', 'N/A')}  ({time.time() - t0:.1f}s)")
            return [avg_metrics]
    except Exception as exc:
        print(f"  Could not parse metrics: {exc}")

    print(f"  Completed in {time.time() - t0:.1f}s (metrics not parsed)")
    return []
