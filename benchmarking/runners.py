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

from models.shared.metrics import average_listed_metrics, cal_od_metrics, compute_metrics
from models.shared.data_load import (
    construct_flat_features, load_graph_data, get_scalers, build_dgl_graph, build_pyg_graph,
    prepare_single_city_flat, prepare_single_city_graph,
)

from .config import DATA_PATH, PROJECT_ROOT, SEED, cleanup_gpu, device


def _predict_flat_array(model, x):
    pred = model.predict(x) if hasattr(model, 'predict') else model(x)
    pred = np.asarray(pred).reshape(-1)
    pred[pred < 0] = 0
    return pred


def _compute_flat_validation_metrics(model, xs_valid, ys_valid, xs_valid_full, ys_valid_full):
    cpc_vals = []
    for x_one, y_one in zip(xs_valid, ys_valid):
        cpc_vals.append(compute_metrics(_predict_flat_array(model, x_one), y_one)['CPC'])

    cpc_fulls = []
    for x_one, y_one in zip(xs_valid_full, ys_valid_full):
        cpc_fulls.append(compute_metrics(_predict_flat_array(model, x_one), y_one)['CPC'])

    return {
        'CPC_val': float(np.mean(cpc_vals)) if cpc_vals else float('nan'),
        'CPC_full': float(np.mean(cpc_fulls)) if cpc_fulls else float('nan'),
    }


def _attach_validation_metrics(metrics_all, validation_metrics):
    for metric in metrics_all:
        metric.update(validation_metrics)
    return metrics_all


def _predict_gmel_matrix(gmel, decoder, nfeat_scaler, nf, adj, dis, device):
    nf_t = torch.FloatTensor(nfeat_scaler.transform(nf)).to(device)
    graph = build_pyg_graph(adj, device)
    with torch.no_grad():
        _, _, _, h_in, h_out = gmel(graph, nf_t)
        h = np.concatenate([h_in.cpu().numpy(), h_out.cpu().numpy()], axis=1)
        n = h.shape[0]
        h_o = h.reshape(n, 1, h.shape[1]).repeat(n, axis=1)
        h_d = h.reshape(1, n, h.shape[1]).repeat(n, axis=0)
        feat = np.concatenate([h_o, h_d, dis.reshape(n, n, 1)], axis=2)
        od_hat = decoder.predict(feat.reshape(-1, h.shape[1] * 2 + 1)).reshape(n, n)
        od_hat[od_hat < 0] = 0
    return od_hat


def _compute_gmel_validation_metrics(gmel, decoder, nfeat_scaler, valid_areas, data_path,
                                     single_city_data=None, device=device):
    if single_city_data is not None:
        pred = _predict_gmel_matrix(
            gmel, decoder, nfeat_scaler,
            single_city_data['nfeat'], single_city_data['adj'], single_city_data['dis'],
            device,
        )
        od = single_city_data['od']
        val_mask = single_city_data['val_mask']
        return {
            'CPC_val': compute_metrics(pred[val_mask], od[val_mask])['CPC'],
            'CPC_full': compute_metrics(pred.ravel(), od.ravel())['CPC'],
        }

    nf_valid, adj_valid, dis_valid, od_valid = load_graph_data(valid_areas, data_path)
    cpcs = []
    for nf, adj, dis, od in zip(nf_valid, adj_valid, dis_valid, od_valid):
        pred = _predict_gmel_matrix(gmel, decoder, nfeat_scaler, nf, adj, dis, device)
        cpcs.append(compute_metrics(pred.ravel(), od.ravel())['CPC'])
    avg_cpc = float(np.mean(cpcs)) if cpcs else float('nan')
    return {'CPC_val': avg_cpc, 'CPC_full': avg_cpc}


def load_model_main(model_name):
    """Dynamically import a model's main.py module."""
    model_dir = PROJECT_ROOT / "models" / model_name
    spec = importlib.util.spec_from_file_location(f"{model_name}_main", model_dir / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enrich_metrics_with_test(mf, pred_matrix, od_matrix, test_mask):
    """Add CPC_nz, CPC_test (and MAE/RMSE variants) to a cal_od_metrics dict."""
    nz = od_matrix > 0
    if np.any(nz):
        mnz = compute_metrics(pred_matrix[nz], od_matrix[nz].astype(float))
        mf['CPC_nz'] = mnz['CPC']
    if test_mask is not None and np.any(test_mask):
        mt = compute_metrics(pred_matrix[test_mask], od_matrix[test_mask].astype(float))
        mf['CPC_test'] = mt['CPC']
        mf['MAE_test'] = mt['MAE']
        mf['RMSE_test'] = mt['RMSE']
    return mf


def run_flat_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    """Train and evaluate a flat-feature model (RF, SVR, GBRT, DGM, GM_E, GM_P)."""
    print(f"\n{'=' * 60}\n  Running: {model_name}\n{'=' * 60}")
    t0 = time.time()
    feature_mode = "gravity" if model_name in ("GM_E", "GM_P") else "full"
    single_city_split = (
        len(train_areas) == len(valid_areas) == len(test_areas) == 1
        and train_areas[0] == valid_areas[0] == test_areas[0]
    )

    od_matrix = None
    test_mask = None
    n_nodes = None

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
        od_matrix = split_data['od_matrix'].astype(float)
        test_mask = split_data['test_mask']
        n_nodes = split_data['n_nodes']
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

    validation_metrics = _compute_flat_validation_metrics(
        model, xs_valid, ys_valid, xs_valid_full, ys_valid_full
    )
    print(f"  Val: CPC_val={validation_metrics['CPC_val']:.4f}  "
          f"CPC_full={validation_metrics['CPC_full']:.4f}")

    del x_train, y_train, xs_valid, ys_valid; gc.collect()

    # Evaluate
    if single_city_split and od_matrix is not None:
        # Single-city: predict full matrix, compute 3-level metrics
        x_full = xs_test[0]
        pred_flat = _predict_flat_array(model, x_full)
        pred_matrix = pred_flat.reshape(n_nodes, n_nodes)
        mf = cal_od_metrics(pred_matrix, od_matrix)
        _enrich_metrics_with_test(mf, pred_matrix, od_matrix, test_mask)
        metrics_all = [mf]
    elif hasattr(module, 'evaluate'):
        metrics_all = module.evaluate(model, xs_test, ys_test)
    else:
        metrics_all = []
        for x_one, y_one in zip(xs_test, ys_test):
            nn = int(np.sqrt(y_one.shape[0]))
            y_hat = model.predict(x_one) if hasattr(model, 'predict') else model(x_one)
            y_hat = y_hat.reshape(nn, nn)
            y_true = y_one.reshape(nn, nn)
            y_hat[y_hat < 0] = 0
            metrics_all.append(cal_od_metrics(y_hat, y_true))

    metrics_all = _attach_validation_metrics(metrics_all, validation_metrics)
    avg = average_listed_metrics(metrics_all)
    print(f"  CPC={avg['CPC']:.4f}  RMSE={avg['RMSE']:.2f}  MAE={avg['MAE']:.2f}  ({time.time() - t0:.1f}s)")
    del model; gc.collect(); cleanup_gpu()
    return metrics_all


def run_graph_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    """Train and evaluate a graph-based model (GMEL variants, NetGAN)."""
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
        if model_name in ("GMEL", "GMEL_GBRT", "GMEL_LGBM"):
            module = load_model_main("GMEL")
            single_city_data = None
            if single_city_split:
                single_city_data = prepare_single_city_graph(train_areas[0], data_path=data_path)
            decoder_type = "lgbm" if model_name == "GMEL_LGBM" else "gbrt"
            encoder_patience = 30 if model_name == "GMEL_LGBM" else 10
            gmel, decoder, nfeat_scaler, dis_scaler = module.train(
                train_areas, valid_areas, str(data_path),
                device=device,
                nfeat_scaler=nfeat_scaler, dis_scaler=dis_scaler, od_scaler=od_scaler,
                single_city_data=single_city_data,
                decoder_type=decoder_type,
                max_epochs=1000, patience=encoder_patience,
            )
            validation_metrics = _compute_gmel_validation_metrics(
                gmel, decoder, nfeat_scaler, valid_areas, data_path,
                single_city_data=single_city_data, device=device,
            )
            print(f"  Val: CPC_val={validation_metrics['CPC_val']:.4f}  "
                  f"CPC_full={validation_metrics['CPC_full']:.4f}")
            if single_city_split and single_city_data is not None:
                od_hat = _predict_gmel_matrix(
                    gmel, decoder, nfeat_scaler,
                    single_city_data['nfeat'], single_city_data['adj'],
                    single_city_data['dis'], device,
                )
                od_full = single_city_data['od'].astype(float)
                mf = cal_od_metrics(od_hat, od_full)
                _enrich_metrics_with_test(mf, od_hat, od_full, single_city_data['test_mask'])
                metrics_all = [mf]
            else:
                nf_test, adj_test, dis_test, od_test = load_graph_data(test_areas, data_path)
                gmel.eval()
                for nf, adj, dis, od in tqdm(zip(nf_test, adj_test, dis_test, od_test),
                                              total=len(nf_test), desc="GMEL eval"):
                    od_hat = _predict_gmel_matrix(gmel, decoder, nfeat_scaler, nf, adj, dis, device)
                    metrics_all.append(cal_od_metrics(od_hat, od))
            metrics_all = _attach_validation_metrics(metrics_all, validation_metrics)
            del gmel, decoder; gc.collect()

        elif model_name == "NetGAN":
            module = load_model_main("NetGAN")
            single_city_data = None
            if single_city_split:
                single_city_data = prepare_single_city_graph(train_areas[0], data_path=data_path)
            trained = module.train(
                train_areas, valid_areas, str(data_path),
                device=device,
                nfeat_scaler=nfeat_scaler, dis_scaler=dis_scaler, od_scaler=od_scaler,
                single_city_data=single_city_data,
            )
            if single_city_split:
                validation_metrics = {'CPC_val': float('nan'), 'CPC_full': float('nan')}
            else:
                val_metrics_list = module.evaluate(trained, valid_areas, str(data_path), device=device)
                val_avg = average_listed_metrics(val_metrics_list) if val_metrics_list else {}
                validation_metrics = {
                    'CPC_val': val_avg.get('CPC', float('nan')),
                    'CPC_full': val_avg.get('CPC', float('nan')),
                }
                print(f"  Val: CPC_val={validation_metrics['CPC_val']:.4f}  "
                      f"CPC_full={validation_metrics['CPC_full']:.4f}")
            if single_city_split and single_city_data is not None:
                # Single-city: predict and compute 3-level metrics
                gen = trained['generator']
                gen.eval()
                nf_s = trained['nfeat_scaler'].transform(single_city_data['nfeat'])
                dis_s = trained['dis_scaler'].transform(
                    single_city_data['dis'].reshape(-1, 1)
                ).reshape(single_city_data['dis'].shape)
                nf_t = torch.FloatTensor(nf_s).to(device)
                dis_t = torch.FloatTensor(dis_s).to(device)
                g = build_dgl_graph(single_city_data['adj'], device)
                with torch.no_grad():
                    od_gen, _, _ = gen.generate_OD_net(g, nf_t, dis_t)
                od_hat = trained['od_scaler'].inverse_transform(
                    od_gen.cpu().numpy().reshape(-1, 1)
                ).reshape(single_city_data['od'].shape)
                od_hat[od_hat < 0] = 0
                od_full = single_city_data['od'].astype(float)
                mf = cal_od_metrics(od_hat, od_full)
                _enrich_metrics_with_test(mf, od_hat, od_full, single_city_data['test_mask'])
                metrics_all = [mf]
            else:
                metrics_all = module.evaluate(trained, test_areas, str(data_path), device=device)
            metrics_all = _attach_validation_metrics(metrics_all, validation_metrics)
            del trained; gc.collect()

        else:
            raise ValueError(f"Unsupported graph model: {model_name}")
    finally:
        cleanup_gpu()

    avg = average_listed_metrics(metrics_all)
    print(f"  CPC={avg['CPC']:.4f}  RMSE={avg['RMSE']:.2f}  MAE={avg['MAE']:.2f}  ({time.time() - t0:.1f}s)")
    return metrics_all


def run_transflower_orig(train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                         gps_loader=None, city_ids=None):
    """Train and evaluate TransFlowerOrig via the GPS training infrastructure."""
    from dataclasses import replace as dc_replace
    from models.GPS.main import train_single_city, train_multi_city
    from models.GPS.config import MC_EPOCHS
    from .config import TRANSFLOWER_ORIG_CONFIG
    from .gps_loader import GPSBenchmarkLoader

    print(f"\n{'=' * 60}\n  Running: TransFlowerOrig\n{'=' * 60}")
    t0 = time.time()

    single_city_split = (
        len(train_areas) == len(valid_areas) == len(test_areas) == 1
        and train_areas[0] == valid_areas[0] == test_areas[0]
    )

    if single_city_split:
        area_id = train_areas[0]
        gps_loader = gps_loader or GPSBenchmarkLoader(
            single_city_id=area_id, data_path=data_path,
        )
        city_data = gps_loader.get_single_city_data(
            pe_type=TRANSFLOWER_ORIG_CONFIG.pe_type, area_id=area_id,
        )
        result = train_single_city(
            "transflower_orig",
            "TransFlower Orig (MLP+TF+RLE)",
            TRANSFLOWER_ORIG_CONFIG,
            city_data=city_data,
        )
    else:
        cfg = dc_replace(TRANSFLOWER_ORIG_CONFIG, mc_epochs=MC_EPOCHS)
        city_ids = city_ids or (train_areas + valid_areas + test_areas)
        # deduplicate while preserving order
        seen = set()
        unique_ids = []
        for cid in city_ids:
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)
        city_ids = unique_ids

        gps_loader = gps_loader or GPSBenchmarkLoader(
            multi_city_ids=city_ids, data_path=data_path,
        )
        mc_dict, mc_train_ids, mc_val_ids, _ = gps_loader.get_multi_city_data(
            pe_type=cfg.pe_type, city_ids=city_ids,
        )
        result = train_multi_city(
            "MC_transflower_orig",
            "MC TransFlower Orig (MLP+TF+RLE)",
            cfg,
            city_data_dict=mc_dict,
            train_city_ids=mc_train_ids,
            val_city_ids=mc_val_ids,
        )

    cleanup_gpu()

    if result.get("status") != "ok":
        print(f"  TransFlowerOrig FAILED: {result.get('status')}")
        return []

    mf = result["metrics_full"]
    # Merge CPC_nz, CPC_test from the GPS result dict
    mnz = result.get("metrics_nonzero", {})
    mt = result.get("metrics_test_pairs", {})
    if mnz:
        mf['CPC_nz'] = mnz.get('CPC', float('nan'))
    if mt:
        mf['CPC_test'] = mt.get('CPC', float('nan'))
        mf['MAE_test'] = mt.get('MAE', float('nan'))
        mf['RMSE_test'] = mt.get('RMSE', float('nan'))

    print(f"  CPC={mf.get('CPC', 'N/A'):.4f}  ({time.time() - t0:.1f}s)")
    return [mf]


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
            avg_metrics.setdefault("CPC_val", float('nan'))
            avg_metrics.setdefault("CPC_full", float('nan'))
            print(f"  CPC={avg_metrics.get('CPC', 'N/A')}  ({time.time() - t0:.1f}s)")
            return [avg_metrics]
    except Exception as exc:
        print(f"  Could not parse metrics: {exc}")

    print(f"  Completed in {time.time() - t0:.1f}s (metrics not parsed)")
    return []
