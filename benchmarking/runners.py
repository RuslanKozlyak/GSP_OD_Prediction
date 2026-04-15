"""Thin dispatcher that calls models/<name>/main.py for training and evaluation."""
import gc
import importlib.util
import json
import subprocess
import sys
import time
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

from models.shared.metrics import (
    _SPLIT_METRIC_NAMES,
    average_listed_metrics,
    average_matrix_split_metrics,
    canonical_od_metrics,
    compute_metrics,
    format_train_val_cpc_metrics,
    masked_split_metrics,
)
from models.shared.data_load import (
    construct_flat_features, load_graph_data, get_scalers, build_dgl_graph, build_pyg_graph,
    prepare_single_city_flat, prepare_single_city_graph,
)

from .artifacts import save_od_artifacts
from .config import (
    DEFAULT_FEATURE_MODE,
    DATA_PATH,
    PROJECT_ROOT,
    WEIGHTS_DIR,
    baseline_multi_city_run_id,
    baseline_single_city_run_id,
    cleanup_gpu,
    device,
    get_baseline_hyperparams,
    set_global_seed,
    SEED,
    FLAT_MULTI_CITY_TRAIN_CAPS,
)


def _predict_flat_array(model, x):
    pred = model.predict(x) if hasattr(model, 'predict') else model(x)
    pred = np.asarray(pred).reshape(-1)
    pred[pred < 0] = 0
    return pred


def _subsample_flat_training(xs, ys, max_samples, seed=SEED):
    total = int(sum(y.shape[0] for y in ys))
    if max_samples is None or max_samples <= 0 or total <= max_samples:
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), total, total

    sizes = np.array([y.shape[0] for y in ys], dtype=np.int64)
    raw = sizes.astype(float) * (float(max_samples) / float(total))
    quotas = np.floor(raw).astype(np.int64)
    if max_samples >= len(sizes):
        quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, sizes)

    remaining = int(max_samples - quotas.sum())
    if remaining > 0:
        order = np.argsort(raw - np.floor(raw))[::-1]
        for idx in order:
            if remaining <= 0:
                break
            add = min(int(sizes[idx] - quotas[idx]), remaining)
            quotas[idx] += add
            remaining -= add

    rng = np.random.default_rng(seed)
    sampled_x, sampled_y = [], []
    for x_one, y_one, quota in zip(xs, ys, quotas):
        quota = int(quota)
        if quota <= 0:
            continue
        if quota >= y_one.shape[0]:
            sampled_x.append(x_one)
            sampled_y.append(y_one)
        else:
            idx = rng.choice(y_one.shape[0], quota, replace=False)
            sampled_x.append(x_one[idx])
            sampled_y.append(y_one[idx])

    sampled = int(sum(y.shape[0] for y in sampled_y))
    return np.concatenate(sampled_x, axis=0), np.concatenate(sampled_y, axis=0), sampled, total


def _flat_pair_counts(areas, data_path):
    root = Path(data_path)
    counts = []
    for area in areas:
        od = np.load(root / area / "od.npy", mmap_mode='r')
        counts.append(int(od.shape[0] * od.shape[1]))
    return np.array(counts, dtype=np.int64)


def _stream_subsample_flat_training(areas, data_path, feature_mode, max_samples, seed=SEED):
    counts = _flat_pair_counts(areas, data_path)
    total = int(counts.sum())
    if max_samples is None or max_samples <= 0 or total <= max_samples:
        xs, ys = construct_flat_features(areas, data_path, feature_mode)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), total, total

    raw = counts.astype(float) * (float(max_samples) / float(total))
    quotas = np.floor(raw).astype(np.int64)
    if max_samples >= len(counts):
        quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, counts)
    remaining = int(max_samples - quotas.sum())
    if remaining > 0:
        order = np.argsort(raw - np.floor(raw))[::-1]
        for idx in order:
            if remaining <= 0:
                break
            add = min(int(counts[idx] - quotas[idx]), remaining)
            quotas[idx] += add
            remaining -= add

    rng = np.random.default_rng(seed)
    sampled_x, sampled_y = [], []
    for area, quota in zip(areas, quotas):
        quota = int(quota)
        if quota <= 0:
            continue
        xs_one, ys_one = construct_flat_features([area], data_path, feature_mode)
        x_one, y_one = xs_one[0], ys_one[0]
        if quota < y_one.shape[0]:
            idx = rng.choice(y_one.shape[0], quota, replace=False)
            x_one = x_one[idx]
            y_one = y_one[idx]
        sampled_x.append(x_one)
        sampled_y.append(y_one)
        del xs_one, ys_one

    sampled = int(sum(y.shape[0] for y in sampled_y))
    return np.concatenate(sampled_x, axis=0), np.concatenate(sampled_y, axis=0), sampled, total


def _compute_flat_train_val_metrics(model, payload):
    if payload['single_city_split']:
        pred_flat = _predict_flat_array(model, payload['x_full'])
        pred_matrix = pred_flat.reshape(payload['n_nodes'], payload['n_nodes'])
        return masked_split_metrics(
            pred_matrix,
            payload['od_matrix'],
            payload['train_mask'],
            payload['val_mask'],
            train_full_mask=payload.get('train_full_mask'),
            val_full_mask=payload.get('val_full_mask'),
        )

    metrics = {}
    for prefix, xs_key, ys_key in (
        ('train', 'xs_train_eval', 'ys_train_eval'),
        ('val', 'xs_valid_full', 'ys_valid_full'),
    ):
        per_city_metrics = []
        if xs_key in payload and ys_key in payload:
            pairs = zip(payload[xs_key], payload[ys_key])
        elif prefix == 'train' and 'train_eval_areas' in payload:
            def _iter_train_pairs():
                for area in payload['train_eval_areas']:
                    xs_one, ys_one = construct_flat_features(
                        [area], payload['data_path'], payload['feature_mode']
                    )
                    yield xs_one[0], ys_one[0]
            pairs = _iter_train_pairs()
        else:
            pairs = []
        for x_one, y_one in pairs:
            n = int(np.sqrt(y_one.shape[0]))
            pred_matrix = _predict_flat_array(model, x_one).reshape(n, n)
            od_matrix = y_one.reshape(n, n)
            one = average_matrix_split_metrics([pred_matrix], [od_matrix], prefix)
            per_city_metrics.append(one)
            del pred_matrix, od_matrix
        if per_city_metrics:
            for key in per_city_metrics[0]:
                vals = [d[key] for d in per_city_metrics]
                metrics[key] = float(np.mean(vals))
        else:
            for m in _SPLIT_METRIC_NAMES:
                metrics[f'{m}_{prefix}_full'] = float('nan')
                metrics[f'{m}_{prefix}_nz'] = float('nan')
    return metrics


def _print_train_val_metrics(metrics):
    if 'CPC_train_full' not in metrics:
        return
    print(f"  Train/Val: {format_train_val_cpc_metrics(metrics)}")


def _print_validation_metrics(metrics):
    cpc_val_nz = metrics.get('CPC_val_nz', float('nan'))
    cpc_val_full = metrics.get('CPC_val_full', float('nan'))
    print(f"  Val: CPC_val_nz={cpc_val_nz:.4f}  CPC_val_full={cpc_val_full:.4f}")


def _attach_validation_metrics(metrics_all, validation_metrics):
    for metric in metrics_all:
        for k, v in validation_metrics.items():
            metric.setdefault(k, v)
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
            'CPC_val_nz': compute_metrics(pred[val_mask], od[val_mask])['CPC'],
            'CPC_val_full': compute_metrics(pred.ravel(), od.ravel())['CPC'],
        }

    nf_valid, adj_valid, dis_valid, od_valid = load_graph_data(
        valid_areas, data_path, feature_mode=DEFAULT_FEATURE_MODE
    )
    cpcs = []
    for nf, adj, dis, od in zip(nf_valid, adj_valid, dis_valid, od_valid):
        pred = _predict_gmel_matrix(gmel, decoder, nfeat_scaler, nf, adj, dis, device)
        cpcs.append(compute_metrics(pred.ravel(), od.ravel())['CPC'])
    avg_cpc = float(np.mean(cpcs)) if cpcs else float('nan')
    return {'CPC_val_nz': avg_cpc, 'CPC_val_full': avg_cpc}


def _compute_gmel_train_val_metrics(gmel, decoder, nfeat_scaler, train_areas, valid_areas,
                                    data_path, single_city_data=None, device=device):
    if single_city_data is not None:
        pred = _predict_gmel_matrix(
            gmel, decoder, nfeat_scaler,
            single_city_data['nfeat'], single_city_data['adj'], single_city_data['dis'],
            device,
        )
        return masked_split_metrics(
            pred,
            single_city_data['od'],
            single_city_data['train_mask'],
            single_city_data['val_mask'],
            train_full_mask=single_city_data.get('train_full_mask'),
            val_full_mask=single_city_data.get('val_full_mask'),
        )

    metrics = {}
    for prefix, areas in (('train', train_areas), ('val', valid_areas)):
        nf_list, adj_list, dis_list, od_list = load_graph_data(
            areas, data_path, feature_mode=DEFAULT_FEATURE_MODE
        )
        pred_matrices = [
            _predict_gmel_matrix(gmel, decoder, nfeat_scaler, nf, adj, dis, device)
            for nf, adj, dis in zip(nf_list, adj_list, dis_list)
        ]
        metrics.update(average_matrix_split_metrics(pred_matrices, od_list, prefix))
    return metrics


def _predict_netgan_matrix(trained, nf, adj, dis, device):
    gen = trained['generator']
    gen.eval()
    nf_s = trained['nfeat_scaler'].transform(nf)
    dis_s = trained['dis_scaler'].transform(dis.reshape(-1, 1)).reshape(dis.shape)
    nf_t = torch.FloatTensor(nf_s).to(device)
    dis_t = torch.FloatTensor(dis_s).to(device)
    graph = build_dgl_graph(adj, device)
    with torch.no_grad():
        od_gen, _, _ = gen.generate_OD_net(graph, nf_t, dis_t)
    od_hat = trained['od_scaler'].inverse_transform(
        od_gen.cpu().numpy().reshape(-1, 1)
    ).reshape(dis.shape)
    od_hat[od_hat < 0] = 0
    return od_hat


def _compute_netgan_train_val_metrics(trained, train_areas, valid_areas, data_path,
                                      single_city_data=None, device=device):
    if single_city_data is not None:
        pred = _predict_netgan_matrix(
            trained,
            single_city_data['nfeat'],
            single_city_data['adj'],
            single_city_data['dis'],
            device,
        )
        return masked_split_metrics(
            pred,
            single_city_data['od'],
            single_city_data['train_mask'],
            single_city_data['val_mask'],
            train_full_mask=single_city_data.get('train_full_mask'),
            val_full_mask=single_city_data.get('val_full_mask'),
        )

    metrics = {}
    for prefix, areas in (('train', train_areas), ('val', valid_areas)):
        nf_list, adj_list, dis_list, od_list = load_graph_data(
            areas, data_path, feature_mode=DEFAULT_FEATURE_MODE
        )
        pred_matrices = [
            _predict_netgan_matrix(trained, nf, adj, dis, device)
            for nf, adj, dis in zip(nf_list, adj_list, dis_list)
        ]
        metrics.update(average_matrix_split_metrics(pred_matrices, od_list, prefix))
    return metrics


def load_model_main(model_name):
    """Dynamically import a model's main.py module."""
    model_dir = PROJECT_ROOT / "models" / model_name
    spec = importlib.util.spec_from_file_location(f"{model_name}_main", model_dir / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_local_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _average_numeric_metrics(metric_dicts):
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


def _is_single_city_split(train_areas, valid_areas, test_areas):
    return (
        len(train_areas) == len(valid_areas) == len(test_areas) == 1
        and train_areas[0] == valid_areas[0] == test_areas[0]
    )


def _benchmark_run_id(model_name, train_areas, valid_areas, test_areas):
    if _is_single_city_split(train_areas, valid_areas, test_areas):
        return baseline_single_city_run_id(model_name, train_areas[0])
    return baseline_multi_city_run_id(model_name)


def _loss_plot_path(run_id):
    return WEIGHTS_DIR.parent / "loss_plots" / f"{run_id}_loss.png"


def _meta_path(run_id):
    return WEIGHTS_DIR / f"{run_id}_meta.joblib"


def _ensure_weights_dir():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_flat_payload(train_areas, valid_areas, test_areas, data_path, feature_mode,
                          train_sample_max=None, keep_train_eval=True,
                          include_train_fit=True):
    single_city_split = _is_single_city_split(train_areas, valid_areas, test_areas)
    payload = {
        'single_city_split': single_city_split,
        'feature_mode': feature_mode,
        'data_path': data_path,
        'od_matrix': None,
        'test_mask': None,
        'n_nodes': None,
        'area_id': None,
        'test_area_ids': list(test_areas),
    }

    if single_city_split:
        split_data = prepare_single_city_flat(
            area_id=train_areas[0], data_path=data_path, feature_mode=feature_mode
        )
        payload.update({
            'x_train': split_data['x_train'],
            'y_train': split_data['y_train'],
            'xs_valid': split_data['xs_val'],
            'ys_valid': split_data['ys_val'],
            'xs_valid_full': split_data['xs_val_full'],
            'ys_valid_full': split_data['ys_val_full'],
            'xs_test': split_data['xs_test'],
            'ys_test': split_data['ys_test'],
            'x_full': split_data['x_full'],
            'y_full': split_data['y_full'],
            'od_matrix': split_data['od_matrix'].astype(float),
            'train_mask': split_data['train_mask'],
            'val_mask': split_data['val_mask'],
            'test_mask': split_data['test_mask'],
            'train_full_mask': split_data['train_full_mask'],
            'val_full_mask': split_data['val_full_mask'],
            'test_full_mask': split_data['test_full_mask'],
            'n_nodes': split_data['n_nodes'],
            'area_id': split_data['area_id'],
            'test_area_ids': [split_data['area_id']],
        })
        return payload

    payload['train_eval_areas'] = list(train_areas)
    if include_train_fit and not keep_train_eval and train_sample_max is not None and train_sample_max > 0:
        x_train, y_train, sampled, total = _stream_subsample_flat_training(
            train_areas, data_path, feature_mode, train_sample_max, seed=SEED
        )
        if sampled < total:
            print(
                f"  Flat train subsample: {sampled:,}/{total:,} pairs "
                f"(cap={train_sample_max:,})"
            )
        payload['x_train'] = x_train
        payload['y_train'] = y_train
    elif include_train_fit or keep_train_eval:
        xs_tr, ys_tr = construct_flat_features(train_areas, data_path, feature_mode)
        if keep_train_eval:
            payload['xs_train_eval'] = xs_tr
            payload['ys_train_eval'] = ys_tr
        if include_train_fit:
            x_train, y_train, sampled, total = _subsample_flat_training(
                xs_tr, ys_tr, train_sample_max, seed=SEED
            )
            if sampled < total:
                print(
                    f"  Flat train subsample: {sampled:,}/{total:,} pairs "
                    f"(cap={train_sample_max:,})"
                )
            payload['x_train'] = x_train
            payload['y_train'] = y_train
        if not keep_train_eval:
            del xs_tr, ys_tr

    xs_valid, ys_valid = construct_flat_features(valid_areas, data_path, feature_mode)
    xs_test, ys_test = construct_flat_features(test_areas, data_path, feature_mode)
    payload.update({
        'xs_valid': xs_valid,
        'ys_valid': ys_valid,
        'xs_valid_full': xs_valid,
        'ys_valid_full': ys_valid,
        'xs_test': xs_test,
        'ys_test': ys_test,
    })
    gc.collect()
    return payload


def _flat_model_suffix(model_name):
    if model_name in ("RF", "SVR", "GBRT"):
        return ".joblib"
    return ".pt"


def _flat_model_path(model_name, run_id):
    return WEIGHTS_DIR / f"{run_id}{_flat_model_suffix(model_name)}"


def _save_flat_model_artifact(module, model_name, run_id, model):
    _ensure_weights_dir()
    path = _flat_model_path(model_name, run_id)
    if hasattr(module, 'save_model'):
        module.save_model(model, str(path))
    else:
        joblib.dump(model, str(path))
    print(f"  -> Weights saved to {path}")
    return path


def _load_flat_model_artifact(module, model_name, run_id):
    path = _flat_model_path(model_name, run_id)
    if not path.exists():
        print(f"  [SKIP] {run_id}: weights not found at {path}")
        return None
    if hasattr(module, 'load_model'):
        return module.load_model(str(path), device=device)
    return joblib.load(str(path))


def _save_gmel_artifacts(run_id, model_name, gmel, decoder, nfeat_scaler, dis_scaler=None):
    _ensure_weights_dir()
    encoder_path = WEIGHTS_DIR / f"{run_id}.pt"
    meta_path = _meta_path(run_id)
    decoder_type = "lgbm" if model_name == "GMEL_LGBM" else "gbrt"

    torch.save({k: v.detach().cpu() for k, v in gmel.state_dict().items()}, encoder_path)
    if decoder_type == "lgbm":
        decoder_path = WEIGHTS_DIR / f"{run_id}_lgbm.lgbm"
        decoder.save_model(str(decoder_path))
    else:
        decoder_path = WEIGHTS_DIR / f"{run_id}_gbrt.joblib"
        joblib.dump(decoder, str(decoder_path))
    joblib.dump(
        {
            'model_name': model_name,
            'decoder_type': decoder_type,
            'nfeat_scaler': nfeat_scaler,
            'dis_scaler': dis_scaler,
            'in_dim': int(getattr(gmel, 'in_dim', 131)),
        },
        str(meta_path),
    )
    print(f"  -> Weights saved to {encoder_path}")
    print(f"  -> Decoder saved to {decoder_path}")
    print(f"  -> Meta saved to {meta_path}")


def _load_gmel_artifacts(run_id):
    encoder_path = WEIGHTS_DIR / f"{run_id}.pt"
    meta_path = _meta_path(run_id)
    if not encoder_path.exists() or not meta_path.exists():
        print(f"  [SKIP] {run_id}: GMEL artifacts not found")
        return None

    meta = joblib.load(str(meta_path))
    decoder_type = meta['decoder_type']
    if decoder_type == "lgbm":
        decoder_path = WEIGHTS_DIR / f"{run_id}_lgbm.lgbm"
        if not decoder_path.exists():
            print(f"  [SKIP] {run_id}: decoder not found at {decoder_path}")
            return None
        import lightgbm as lgb
        decoder = lgb.Booster(model_file=str(decoder_path))
    else:
        decoder_path = WEIGHTS_DIR / f"{run_id}_gbrt.joblib"
        if not decoder_path.exists():
            print(f"  [SKIP] {run_id}: decoder not found at {decoder_path}")
            return None
        decoder = joblib.load(str(decoder_path))

    gmel_module = _load_local_module("benchmark_gmel_model", PROJECT_ROOT / "models" / "GMEL" / "model.py")
    gmel = gmel_module.GMEL(in_dim=int(meta.get('in_dim', 131))).to(device)
    gmel.load_state_dict(torch.load(str(encoder_path), map_location=device, weights_only=False))
    gmel.eval()
    return gmel, decoder, meta['nfeat_scaler'], meta.get('dis_scaler')


def _save_netgan_artifacts(run_id, trained):
    _ensure_weights_dir()
    generator_path = WEIGHTS_DIR / f"{run_id}.pt"
    meta_path = _meta_path(run_id)
    torch.save(
        {k: v.detach().cpu() for k, v in trained['generator'].state_dict().items()},
        generator_path,
    )
    joblib.dump(
        {
            'nfeat_scaler': trained['nfeat_scaler'],
            'dis_scaler': trained['dis_scaler'],
            'od_scaler': trained['od_scaler'],
        },
        str(meta_path),
    )
    print(f"  -> Weights saved to {generator_path}")
    print(f"  -> Meta saved to {meta_path}")


def _load_netgan_artifacts(run_id):
    generator_path = WEIGHTS_DIR / f"{run_id}.pt"
    meta_path = _meta_path(run_id)
    if not generator_path.exists() or not meta_path.exists():
        print(f"  [SKIP] {run_id}: NetGAN artifacts not found")
        return None

    netgan_module = _load_local_module("benchmark_netgan_model", PROJECT_ROOT / "models" / "NetGAN" / "model.py")
    generator = netgan_module.Generator().to(device)
    generator.load_state_dict(torch.load(str(generator_path), map_location=device, weights_only=False))
    generator.eval()
    trained = joblib.load(str(meta_path))
    trained['generator'] = generator
    return trained


def train_flat_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                     run_id=None):
    """Train a flat-feature baseline and persist its weights."""
    print(f"\n{'=' * 60}\n  Training: {model_name}\n{'=' * 60}")
    set_global_seed()
    run_id = run_id or _benchmark_run_id(model_name, train_areas, valid_areas, test_areas)
    feature_mode = "gravity" if model_name in ("GM_E", "GM_P") else DEFAULT_FEATURE_MODE
    multi_city_split = not _is_single_city_split(train_areas, valid_areas, test_areas)
    cap_value = FLAT_MULTI_CITY_TRAIN_CAPS.get(model_name) if multi_city_split else None
    capped = cap_value is not None and cap_value > 0
    payload = _prepare_flat_payload(
        train_areas,
        valid_areas,
        test_areas,
        data_path,
        feature_mode,
        train_sample_max=cap_value if capped else None,
        keep_train_eval=not capped,
    )
    module = load_model_main(model_name)
    hp = get_baseline_hyperparams(model_name)
    model = None
    try:
        if not hasattr(module, 'train'):
            raise ValueError(f"Model {model_name} has no train() function")
        split_metric_kwargs = {}
        if model_name in ("DGM", "GM_E", "GM_P"):
            if payload.get('single_city_split'):
                split_metric_kwargs = {
                    'x_full': payload.get('x_full'),
                    'od_matrix': payload.get('od_matrix'),
                    'train_mask': payload.get('train_mask'),
                    'val_mask': payload.get('val_mask'),
                    'train_full_mask': payload.get('train_full_mask'),
                    'val_full_mask': payload.get('val_full_mask'),
                }
            elif 'xs_train_eval' in payload and 'ys_train_eval' in payload:
                split_metric_kwargs = {
                    'xs_train_full': payload['xs_train_eval'],
                    'ys_train_full': payload['ys_train_eval'],
                }
        model = module.train(
            payload['x_train'],
            payload['y_train'],
            payload['xs_valid'],
            payload['ys_valid'],
            xs_valid_full=payload['xs_valid_full'],
            ys_valid_full=payload['ys_valid_full'],
            device=device,
            batch_size=hp.get('batch_size', 10_000),
            max_epochs=hp.get('max_epochs', 10_000),
            patience=hp.get('patience', 100),
            loss_plot_path=_loss_plot_path(run_id),
            **split_metric_kwargs,
            **{k: v for k, v in hp.items() if k not in ('batch_size', 'max_epochs', 'patience')},
        )
        validation_metrics = _compute_flat_train_val_metrics(model, payload)
        _print_validation_metrics(validation_metrics)
        _print_train_val_metrics(validation_metrics)
        _save_flat_model_artifact(module, model_name, run_id, model)
        return run_id
    finally:
        del payload
        if model is not None:
            del model
        gc.collect()
        cleanup_gpu()


def infer_flat_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                     run_id=None):
    """Load a flat-feature baseline from disk and evaluate it."""
    print(f"\n{'=' * 60}\n  Loading: {model_name}\n{'=' * 60}")
    t0 = time.time()
    run_id = run_id or _benchmark_run_id(model_name, train_areas, valid_areas, test_areas)
    feature_mode = "gravity" if model_name in ("GM_E", "GM_P") else DEFAULT_FEATURE_MODE
    multi_city_split = not _is_single_city_split(train_areas, valid_areas, test_areas)
    cap_value = FLAT_MULTI_CITY_TRAIN_CAPS.get(model_name) if multi_city_split else None
    capped = cap_value is not None and cap_value > 0
    payload = _prepare_flat_payload(
        train_areas,
        valid_areas,
        test_areas,
        data_path,
        feature_mode,
        keep_train_eval=not capped,
        include_train_fit=False,
    )
    module = load_model_main(model_name)
    model = None
    try:
        model = _load_flat_model_artifact(module, model_name, run_id)
        if model is None:
            return []
        validation_metrics = _compute_flat_train_val_metrics(model, payload)
        _print_validation_metrics(validation_metrics)
        _print_train_val_metrics(validation_metrics)

        if payload['single_city_split'] and payload['od_matrix'] is not None:
            x_full = payload['xs_test'][0]
            pred_flat = _predict_flat_array(model, x_full)
            pred_matrix = pred_flat.reshape(payload['n_nodes'], payload['n_nodes'])
            save_od_artifacts(
                run_id,
                pred_matrix,
                payload['od_matrix'],
                city_id=payload['area_id'],
                model_name=model_name,
            )
            mf = canonical_od_metrics(
                pred_matrix,
                payload['od_matrix'],
                test_mask=payload['test_mask'],
                test_full_mask=payload.get('test_full_mask'),
                train_mask=payload.get('train_mask'),
                val_mask=payload.get('val_mask'),
                train_full_mask=payload.get('train_full_mask'),
                val_full_mask=payload.get('val_full_mask'),
            )
            metric_runs = _attach_validation_metrics([mf], validation_metrics)
        else:
            metrics_all = []
            for area_id, x_one, y_one in zip(
                payload['test_area_ids'], payload['xs_test'], payload['ys_test']
            ):
                nn = int(np.sqrt(y_one.shape[0]))
                y_hat = _predict_flat_array(model, x_one).reshape(nn, nn)
                y_true = y_one.reshape(nn, nn)
                save_od_artifacts(
                    run_id,
                    y_hat,
                    y_true,
                    city_id=area_id,
                    model_name=model_name,
                )
                metrics_all.append(canonical_od_metrics(y_hat, y_true))
            metric_runs = _attach_validation_metrics(metrics_all, validation_metrics)

        if metric_runs:
            avg = average_listed_metrics(metric_runs)
            print(f"  CPC_full={avg['CPC_full']:.4f}  RMSE_full={avg['RMSE_full']:.2f}  "
                  f"MAE_full={avg['MAE_full']:.2f}  ({time.time() - t0:.1f}s)")
        return metric_runs
    finally:
        del payload
        if model is not None:
            del model
        gc.collect()
        cleanup_gpu()


def run_flat_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    """Train, save, then load+evaluate a flat-feature model."""
    run_id = train_flat_model(model_name, train_areas, valid_areas, test_areas, data_path)
    return infer_flat_model(
        model_name, train_areas, valid_areas, test_areas, data_path,
        run_id=run_id,
    )


def train_graph_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                      run_id=None):
    """Train a graph-based baseline and persist its weights."""
    print(f"\n{'=' * 60}\n  Training: {model_name}\n{'=' * 60}")
    set_global_seed()
    run_id = run_id or _benchmark_run_id(model_name, train_areas, valid_areas, test_areas)
    single_city_split = _is_single_city_split(train_areas, valid_areas, test_areas)

    if single_city_split:
        nfeat_scaler = dis_scaler = od_scaler = None
    else:
        nf_train, _, dis_train, od_train = load_graph_data(
            train_areas, data_path, feature_mode=DEFAULT_FEATURE_MODE
        )
        nfeat_scaler, dis_scaler, od_scaler = get_scalers(nf_train, dis_train, od_train)

    try:
        if model_name in ("GMEL", "GMEL_GBRT", "GMEL_LGBM"):
            module = load_model_main("GMEL")
            single_city_data = None
            if single_city_split:
                single_city_data = prepare_single_city_graph(
                    train_areas[0], data_path=data_path, feature_mode=DEFAULT_FEATURE_MODE
                )
            hp = get_baseline_hyperparams(model_name)
            gmel, decoder, nfeat_scaler, dis_scaler = module.train(
                train_areas, valid_areas, str(data_path),
                device=device,
                nfeat_scaler=nfeat_scaler, dis_scaler=dis_scaler, od_scaler=od_scaler,
                single_city_data=single_city_data,
                feature_mode=DEFAULT_FEATURE_MODE,
                decoder_type=hp.get('decoder_type', 'gbrt'),
                max_epochs=hp.get('encoder_max_epochs', 10_000),
                patience=hp.get('encoder_patience', 100),
                encoder_lr=hp.get('encoder_lr', 3e-4),
                loss_plot_path=_loss_plot_path(run_id),
                **{k: v for k, v in hp.items()
                   if k not in ('decoder_type', 'encoder_max_epochs', 'encoder_patience', 'encoder_lr')},
            )
            validation_metrics = _compute_gmel_validation_metrics(
                gmel, decoder, nfeat_scaler, valid_areas, data_path,
                single_city_data=single_city_data, device=device,
            )
            validation_metrics.update(_compute_gmel_train_val_metrics(
                gmel, decoder, nfeat_scaler, train_areas, valid_areas, data_path,
                single_city_data=single_city_data, device=device,
            ))
            _print_validation_metrics(validation_metrics)
            _print_train_val_metrics(validation_metrics)
            _save_gmel_artifacts(run_id, model_name, gmel, decoder, nfeat_scaler, dis_scaler)
            del gmel, decoder
            return run_id

        if model_name == "NetGAN":
            module = load_model_main("NetGAN")
            single_city_data = None
            if single_city_split:
                single_city_data = prepare_single_city_graph(
                    train_areas[0], data_path=data_path, feature_mode=DEFAULT_FEATURE_MODE
                )
            hp = get_baseline_hyperparams(model_name)
            trained = module.train(
                train_areas, valid_areas, str(data_path),
                device=device,
                nfeat_scaler=nfeat_scaler, dis_scaler=dis_scaler, od_scaler=od_scaler,
                single_city_data=single_city_data,
                feature_mode=DEFAULT_FEATURE_MODE,
                n_epochs=hp.get('n_epochs', 2),
                lr=hp.get('lr', 3e-4),
                gp_lambda=hp.get('gp_lambda', 10),
                batch_size=hp.get('batch_size', 128),
                verbose=hp.get('verbose', 1),
            )
            train_val_metrics = _compute_netgan_train_val_metrics(
                trained, train_areas, valid_areas, data_path,
                single_city_data=single_city_data, device=device,
            )
            _print_train_val_metrics(train_val_metrics)
            _save_netgan_artifacts(run_id, trained)
            del trained
            return run_id

        raise ValueError(f"Unsupported graph model: {model_name}")
    finally:
        gc.collect()
        cleanup_gpu()


def infer_graph_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                      run_id=None):
    """Load a graph-based baseline from disk and evaluate it."""
    print(f"\n{'=' * 60}\n  Loading: {model_name}\n{'=' * 60}")
    t0 = time.time()
    run_id = run_id or _benchmark_run_id(model_name, train_areas, valid_areas, test_areas)
    single_city_split = _is_single_city_split(train_areas, valid_areas, test_areas)

    try:
        if model_name in ("GMEL", "GMEL_GBRT", "GMEL_LGBM"):
            loaded = _load_gmel_artifacts(run_id)
            if loaded is None:
                return []
            gmel, decoder, nfeat_scaler, _ = loaded
            single_city_data = None
            if single_city_split:
                single_city_data = prepare_single_city_graph(
                    train_areas[0], data_path=data_path, feature_mode=DEFAULT_FEATURE_MODE
                )
            validation_metrics = _compute_gmel_validation_metrics(
                gmel, decoder, nfeat_scaler, valid_areas, data_path,
                single_city_data=single_city_data, device=device,
            )
            validation_metrics.update(_compute_gmel_train_val_metrics(
                gmel, decoder, nfeat_scaler, train_areas, valid_areas, data_path,
                single_city_data=single_city_data, device=device,
            ))
            _print_validation_metrics(validation_metrics)
            _print_train_val_metrics(validation_metrics)

            if single_city_split and single_city_data is not None:
                od_hat = _predict_gmel_matrix(
                    gmel, decoder, nfeat_scaler,
                    single_city_data['nfeat'], single_city_data['adj'],
                    single_city_data['dis'], device,
                )
                od_full = single_city_data['od'].astype(float)
                save_od_artifacts(
                    run_id,
                    od_hat,
                    od_full,
                    city_id=single_city_data['area_id'],
                    model_name=model_name,
                )
                mf = canonical_od_metrics(
                    od_hat,
                    od_full,
                    test_mask=single_city_data['test_mask'],
                    test_full_mask=single_city_data.get('test_full_mask'),
                    train_mask=single_city_data.get('train_mask'),
                    val_mask=single_city_data.get('val_mask'),
                    train_full_mask=single_city_data.get('train_full_mask'),
                    val_full_mask=single_city_data.get('val_full_mask'),
                )
                metric_runs = _attach_validation_metrics([mf], validation_metrics)
            else:
                metrics_all = []
                nf_test, adj_test, dis_test, od_test = load_graph_data(
                    test_areas, data_path, feature_mode=DEFAULT_FEATURE_MODE
                )
                gmel.eval()
                for area_id, nf, adj, dis, od in tqdm(
                    zip(test_areas, nf_test, adj_test, dis_test, od_test),
                    total=len(nf_test),
                    desc="GMEL eval",
                ):
                    od_hat = _predict_gmel_matrix(gmel, decoder, nfeat_scaler, nf, adj, dis, device)
                    save_od_artifacts(
                        run_id,
                        od_hat,
                        od.astype(float),
                        city_id=area_id,
                        model_name=model_name,
                    )
                    metrics_all.append(canonical_od_metrics(od_hat, od))
                metric_runs = _attach_validation_metrics(metrics_all, validation_metrics)
            del gmel, decoder

        elif model_name == "NetGAN":
            trained = _load_netgan_artifacts(run_id)
            if trained is None:
                return []
            module = load_model_main("NetGAN")
            single_city_data = None
            if single_city_split:
                single_city_data = prepare_single_city_graph(
                    train_areas[0], data_path=data_path, feature_mode=DEFAULT_FEATURE_MODE
                )
                validation_metrics = {'CPC_val_nz': float('nan'), 'CPC_val_full': float('nan')}
            else:
                val_metrics_list = module.evaluate(
                    trained, valid_areas, str(data_path),
                    device=device, feature_mode=DEFAULT_FEATURE_MODE,
                )
                val_avg = average_listed_metrics(val_metrics_list) if val_metrics_list else {}
                validation_metrics = {
                    'CPC_val_nz': val_avg.get('CPC_full', float('nan')),
                    'CPC_val_full': val_avg.get('CPC_full', float('nan')),
                }
            validation_metrics.update(_compute_netgan_train_val_metrics(
                trained, train_areas, valid_areas, data_path,
                single_city_data=single_city_data, device=device,
            ))
            _print_validation_metrics(validation_metrics)
            _print_train_val_metrics(validation_metrics)

            if single_city_split and single_city_data is not None:
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
                save_od_artifacts(
                    run_id,
                    od_hat,
                    od_full,
                    city_id=single_city_data['area_id'],
                    model_name=model_name,
                )
                mf = canonical_od_metrics(
                    od_hat,
                    od_full,
                    test_mask=single_city_data['test_mask'],
                    test_full_mask=single_city_data.get('test_full_mask'),
                    train_mask=single_city_data.get('train_mask'),
                    val_mask=single_city_data.get('val_mask'),
                    train_full_mask=single_city_data.get('train_full_mask'),
                    val_full_mask=single_city_data.get('val_full_mask'),
                )
                metric_runs = _attach_validation_metrics([mf], validation_metrics)
            else:
                metrics_all = []
                nf_test, adj_test, dis_test, od_test = load_graph_data(
                    test_areas, data_path, feature_mode=DEFAULT_FEATURE_MODE
                )
                gen = trained['generator']
                gen.eval()
                for area_id, nf, adj, dis, od in zip(
                    test_areas, nf_test, adj_test, dis_test, od_test
                ):
                    nf_s = trained['nfeat_scaler'].transform(nf)
                    dis_s = trained['dis_scaler'].transform(
                        dis.reshape(-1, 1)
                    ).reshape(dis.shape)
                    nf_t = torch.FloatTensor(nf_s).to(device)
                    dis_t = torch.FloatTensor(dis_s).to(device)
                    g = build_dgl_graph(adj, device)
                    with torch.no_grad():
                        od_gen, _, _ = gen.generate_OD_net(g, nf_t, dis_t)
                    od_hat = trained['od_scaler'].inverse_transform(
                        od_gen.cpu().numpy().reshape(-1, 1)
                    ).reshape(od.shape)
                    od_hat[od_hat < 0] = 0
                    save_od_artifacts(
                        run_id,
                        od_hat,
                        od.astype(float),
                        city_id=area_id,
                        model_name=model_name,
                    )
                    metrics_all.append(canonical_od_metrics(od_hat, od))
                metric_runs = _attach_validation_metrics(metrics_all, validation_metrics)
            del trained

        else:
            raise ValueError(f"Unsupported graph model: {model_name}")

        if metric_runs:
            avg = average_listed_metrics(metric_runs)
            print(f"  CPC_full={avg['CPC_full']:.4f}  RMSE_full={avg['RMSE_full']:.2f}  "
                  f"MAE_full={avg['MAE_full']:.2f}  ({time.time() - t0:.1f}s)")
        return metric_runs
    finally:
        gc.collect()
        cleanup_gpu()


def run_graph_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH,
):
    """Train, save, then load+evaluate a graph-based model."""
    run_id = train_graph_model(model_name, train_areas, valid_areas, test_areas, data_path)
    return infer_graph_model(
        model_name, train_areas, valid_areas, test_areas, data_path,
        run_id=run_id,
    )


def _gps_baseline_labels(model_name):
    if model_name == "TransFlowerOrig":
        return (
            "TransFlower Orig (MLP+TF+RLE)",
            "MC TransFlower Orig (MLP+TF+RLE)",
        )
    if model_name == "GAT_GAN_Orig":
        return (
            "Orig GAT-GAN (GAT+Linear+WGAN-clip)",
            "MC Orig GAT-GAN (GAT+Linear+WGAN-clip)",
        )
    if model_name == "ODGN":
        return (
            "ODGN (GAT+Gravity-Guided+WGAN-clip)",
            "MC ODGN (GAT+Gravity-Guided+WGAN-clip)",
        )
    raise ValueError(f"Unsupported GPS baseline model: {model_name}")


def train_gps_baseline(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                       gps_loader=None, city_ids=None, run_id=None):
    """Train and persist a GPS-family baseline via the GPS training infrastructure."""
    from dataclasses import replace as dc_replace
    from models.GPS.main import train_single_city, train_multi_city
    from models.GPS.config import MC_EPOCHS
    from .config import GPS_BASELINE_CONFIGS
    from .gps_loader import GPSBenchmarkLoader

    cfg_template = GPS_BASELINE_CONFIGS.get(model_name)
    if cfg_template is None:
        raise ValueError(f"Unsupported GPS baseline model: {model_name}")
    sc_label, mc_label = _gps_baseline_labels(model_name)

    print(f"\n{'=' * 60}\n  Training: {model_name}\n{'=' * 60}")
    set_global_seed()
    run_id = run_id or _benchmark_run_id(model_name, train_areas, valid_areas, test_areas)
    single_city_split = _is_single_city_split(train_areas, valid_areas, test_areas)

    if single_city_split:
        area_id = train_areas[0]
        gps_loader = gps_loader or GPSBenchmarkLoader(
            single_city_id=area_id, data_path=data_path,
        )
        city_data = gps_loader.get_single_city_data(
            pe_type=cfg_template.pe_type,
            area_id=area_id,
            pair_split_mode=cfg_template.pair_split_mode,
        )
        result = train_single_city(
            run_id,
            sc_label,
            cfg_template,
            city_data=city_data,
        )
    else:
        cfg = dc_replace(cfg_template, mc_epochs=MC_EPOCHS)
        city_ids = city_ids or (train_areas + valid_areas + test_areas)
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
        mc_dict, mc_train_ids, mc_val_ids, mc_test_ids = gps_loader.get_multi_city_data(
            pe_type=cfg.pe_type,
            city_ids=city_ids,
            pair_split_mode=cfg.pair_split_mode,
        )
        result = train_multi_city(
            run_id,
            mc_label,
            cfg,
            city_data_dict=mc_dict,
            train_city_ids=mc_train_ids,
            val_city_ids=mc_val_ids,
            test_city_ids=mc_test_ids,
        )

    cleanup_gpu()
    if result.get("status") != "ok":
        print(f"  {model_name} FAILED: {result.get('status')}")
        return None
    return run_id


def infer_gps_baseline(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                       gps_loader=None, city_ids=None, run_id=None):
    """Load a persisted GPS-family baseline run and evaluate it."""
    from .gps_loader import GPSBenchmarkLoader

    print(f"\n{'=' * 60}\n  Loading: {model_name}\n{'=' * 60}")
    t0 = time.time()
    run_id = run_id or _benchmark_run_id(model_name, train_areas, valid_areas, test_areas)
    single_city_split = _is_single_city_split(train_areas, valid_areas, test_areas)
    metric_runs = []

    if single_city_split:
        area_id = train_areas[0]
        gps_loader = gps_loader or GPSBenchmarkLoader(
            single_city_id=area_id, data_path=data_path,
        )
        metrics = gps_loader.load_gps_results(
            run_id,
            area_id=area_id,
        )
        if metrics:
            metric_runs.append(metrics)
    else:
        city_ids = city_ids or (train_areas + valid_areas + test_areas)
        city_ids = list(dict.fromkeys(city_ids))
        gps_loader = gps_loader or GPSBenchmarkLoader(
            multi_city_ids=city_ids, data_path=data_path,
        )
        metric_groups = gps_loader.load_multi_city_gps_results(
            run_id,
            city_ids=city_ids,
            evaluate_all_cities=True,
            return_split_groups=True,
        )
        if metric_groups and metric_groups.get("all"):
            averaged = _average_numeric_metrics(metric_groups["all"])
            test_metrics = metric_groups.get("test") or []
            if test_metrics:
                averaged_test = _average_numeric_metrics(test_metrics)
                for key in averaged_test:
                    if (
                        key.endswith("_test_full")
                        or key.endswith("_test_nz")
                    ):
                        averaged[key] = averaged_test[key]
            for prefix, split_metrics in (
                ("train", metric_groups.get("train") or []),
                ("val", metric_groups.get("val") or []),
            ):
                if split_metrics:
                    averaged_split = _average_numeric_metrics(split_metrics)
                    for metric_name in ("CPC", "MAE", "RMSE", "MAPE", "SMAPE", "NRMSE"):
                        for variant in ("full", "nz"):
                            source_key = f"{metric_name}_{variant}"
                            if source_key in averaged_split:
                                averaged[f"{metric_name}_{prefix}_{variant}"] = (
                                    averaged_split[source_key]
                                )
            metric_runs.append(averaged)

    if metric_runs:
        avg = average_listed_metrics(metric_runs)
        print(f"  CPC_full={avg.get('CPC_full', float('nan')):.4f}  ({time.time() - t0:.1f}s)")
        if "CPC_train_full" in avg:
            print(f"  Train/Val: {format_train_val_cpc_metrics(avg)}")
    return metric_runs


def train_transflower_orig(train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                           gps_loader=None, city_ids=None, run_id=None):
    """Backward-compatible wrapper for the TransFlower GPS-family baseline."""
    return train_gps_baseline(
        "TransFlowerOrig",
        train_areas, valid_areas, test_areas, data_path,
        gps_loader=gps_loader, city_ids=city_ids, run_id=run_id,
    )


def infer_transflower_orig(train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                           gps_loader=None, city_ids=None, run_id=None):
    """Backward-compatible wrapper for the TransFlower GPS-family baseline."""
    return infer_gps_baseline(
        "TransFlowerOrig",
        train_areas, valid_areas, test_areas, data_path,
        gps_loader=gps_loader, city_ids=city_ids, run_id=run_id,
    )


def run_transflower_orig(train_areas, valid_areas, test_areas, data_path=DATA_PATH,
                         gps_loader=None, city_ids=None):
    """Train, save, then load+evaluate TransFlowerOrig."""
    run_id = train_transflower_orig(
        train_areas, valid_areas, test_areas, data_path,
        gps_loader=gps_loader, city_ids=city_ids,
    )
    if run_id is None:
        return []
    return infer_transflower_orig(
        train_areas, valid_areas, test_areas, data_path,
        gps_loader=gps_loader, city_ids=city_ids,
        run_id=run_id,
    )


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
            for old, new in (
                ("CPC", "CPC_full"),
                ("MAE", "MAE_full"),
                ("RMSE", "RMSE_full"),
                ("CPC_nonzero", "CPC_nz"),
                ("MAE_nonzero", "MAE_nz"),
                ("RMSE_nonzero", "RMSE_nz"),
                ("CPC_test", "CPC_test_nz"),
                ("MAE_test", "MAE_test_nz"),
                ("RMSE_test", "RMSE_test_nz"),
                ("MAPE_test", "MAPE_test_nz"),
                ("SMAPE_test", "SMAPE_test_nz"),
                ("NRMSE_test", "NRMSE_test_nz"),
            ):
                if old in avg_metrics and new not in avg_metrics:
                    avg_metrics[new] = avg_metrics.pop(old)
            avg_metrics.setdefault("CPC_val_nz", float('nan'))
            for key in ("CPC_train_full", "CPC_val_full", "CPC_train_nz", "CPC_val_nz"):
                avg_metrics.setdefault(key, float('nan'))
            print(f"  CPC_full={avg_metrics.get('CPC_full', 'N/A')}  ({time.time() - t0:.1f}s)")
            return [dict(avg_metrics)]
    except Exception as exc:
        print(f"  Could not parse metrics: {exc}")

    print(f"  Completed in {time.time() - t0:.1f}s (metrics not parsed)")
    return []
