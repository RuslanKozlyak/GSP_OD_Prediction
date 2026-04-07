import time
import gc

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from models.shared.metrics import cal_od_metrics, compute_metrics
from models.shared.plotting import save_loss_plot


def _masked_mse(pred, target, mask=None):
    if mask is None:
        return torch.mean((pred - target) ** 2)
    if mask.sum().item() == 0:
        return pred.new_tensor(0.0)
    return torch.mean((pred[mask] - target[mask]) ** 2)


def _marginal_mse(pred, target):
    # Keep marginals as 1D vectors; otherwise (N, 1) vs (N,) broadcasts to (N, N).
    if pred.dim() == target.dim() + 1 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if pred.shape != target.shape:
        raise ValueError(f"Marginal shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    return torch.mean((pred - target) ** 2)


def _pair_embeddings_to_features(h_in, h_out, dis):
    h = np.concatenate([h_in, h_out], axis=1)
    n = h.shape[0]
    h_o = h.reshape(n, 1, h.shape[1]).repeat(n, axis=1)
    h_d = h.reshape(1, n, h.shape[1]).repeat(n, axis=0)
    return np.concatenate([h_o, h_d, dis.reshape(n, n, 1)], axis=2).reshape(-1, h.shape[1] * 2 + 1)


def _transform_masked_matrix(matrix, scaler, mask):
    scaled = np.zeros_like(matrix, dtype=np.float32)
    if mask is not None and mask.any():
        scaled[mask] = scaler.transform(matrix[mask].reshape(-1, 1)).reshape(-1)
    return scaled


def _predict_bilinear_matrix(gmel, nf_t, g, od_scaler):
    gmel.eval()
    with torch.no_grad():
        _, _, flow, _, _ = gmel(g, nf_t)
    pred = od_scaler.inverse_transform(flow.detach().cpu().numpy().reshape(-1, 1)).reshape(flow.shape)
    pred[pred < 0] = 0
    return pred


def _predict_decoder_matrix(decoder, h_in, h_out, dis):
    pred = decoder.predict(_pair_embeddings_to_features(h_in, h_out, dis)).reshape(dis.shape[0], dis.shape[1])
    pred[pred < 0] = 0
    return pred


def _print_stage_metrics(stage_name, pred, od_np, test_mask=None):
    nz = od_np > 0
    mf = cal_od_metrics(pred, od_np)
    mnz = compute_metrics(pred[nz], od_np[nz]) if np.any(nz) else {'CPC': 0.0, 'MAE': 0.0, 'RMSE': 0.0}
    mt = (
        compute_metrics(pred[test_mask], od_np[test_mask])
        if test_mask is not None and np.any(test_mask)
        else None
    )
    print(f"\n  === {stage_name} ===")
    if mt is not None:
        print(f"    CPC_full={mf['CPC']:.4f}  CPC_nz={mnz['CPC']:.4f}  "
              f"CPC_test={mt['CPC']:.4f}  MAE={mf['MAE']:.4f}  RMSE={mf['RMSE']:.4f}")
    else:
        print(f"    CPC_full={mf['CPC']:.4f}  CPC_nz={mnz['CPC']:.4f}  "
              f"MAE={mf['MAE']:.4f}  RMSE={mf['RMSE']:.4f}")
    print(f"    Full metrics: {mf}")
    print(f"    Nonzero metrics: {mnz}")
    if mt is not None:
        print(f"    Test-pair metrics: {mt}")
    return mf, mnz, mt


def _average_metric_dicts(metrics):
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}


def _print_averaged_stage_metrics(stage_name, metrics_full, metrics_nonzero):
    mf = _average_metric_dicts(metrics_full)
    mnz = _average_metric_dicts(metrics_nonzero)
    print(f"\n  === {stage_name} ===")
    print(f"    CPC_full={mf.get('CPC', float('nan')):.4f}  "
          f"CPC_nz={mnz.get('CPC', float('nan')):.4f}  "
          f"MAE={mf.get('MAE', float('nan')):.4f}  "
          f"RMSE={mf.get('RMSE', float('nan')):.4f}")
    print(f"    Avg full metrics: {mf}")
    print(f"    Avg nonzero metrics: {mnz}")
    return mf, mnz


def _fit_decoder(decoder_type, x_train, y_train, x_val=None, y_val=None, **kwargs):
    verbose = int(kwargs.get('verbose', 0) or 0)
    if decoder_type == 'lgbm':
        import lightgbm as lgb

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': kwargs.get('lgbm_learning_rate', 0.05),
            'num_leaves': kwargs.get('lgbm_num_leaves', 63),
            'max_depth': kwargs.get('lgbm_max_depth', 8),
            'subsample': kwargs.get('lgbm_subsample', 0.8),
            'colsample_bytree': kwargs.get('lgbm_colsample_bytree', 0.8),
            'verbosity': verbose if verbose else -1,
            'seed': 42,
        }
        num_boost_round = kwargs.get('lgbm_num_boost_round', 1000)
        early_stopping_rounds = kwargs.get('lgbm_early_stopping', 50)
        log_period = kwargs.get('lgbm_log_period', 100) if verbose else 0
        train_set = lgb.Dataset(x_train, y_train)
        if x_val is not None and len(x_val) > 0:
            valid_set = lgb.Dataset(x_val, y_val, reference=train_set)
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=bool(verbose))]
            if log_period:
                callbacks.append(lgb.log_evaluation(log_period))
            return lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=[valid_set],
                callbacks=callbacks,
            )
        return lgb.train(params, train_set, num_boost_round=num_boost_round)

    decoder = GradientBoostingRegressor(
        n_estimators=kwargs.get('gbrt_n_estimators', 20),
        min_samples_split=kwargs.get('gbrt_min_samples_split', 2),
        min_samples_leaf=kwargs.get('gbrt_min_samples_leaf', 2),
        max_depth=None,
        verbose=verbose,
    )
    decoder.fit(x_train, y_train)
    return decoder


def train(train_areas, val_areas, data_path,
          device=None, nfeat_scaler=None, dis_scaler=None, od_scaler=None,
          max_epochs=1000, patience=10, single_city_data=None, decoder_type='gbrt',
          encoder_lr=3e-4, loss_plot_path=None, verbose=1, **decoder_kwargs):
    """Train GMEL (PyG GAT encoder + tree decoder).

    Args:
        train_areas: list of area IDs for training
        val_areas:   list of area IDs for validation
        data_path:   path to data root directory
        device:      torch.device
        nfeat_scaler, dis_scaler, od_scaler: pre-fitted sklearn scalers
            (if None, fitted on train_areas data)
        max_epochs / patience / encoder_lr: GAT training schedule
        verbose: enables tqdm, decoder logs, and stage metric prints
        single_city_data: optional dict from prepare_single_city_graph() for
            honest train/val masking inside one city

    Returns:
        (gmel_net, decoder, nfeat_scaler, dis_scaler)
    """
    import os, sys
    sys.modules.pop('model', None)    # prevent collision when run after another model
    sys.modules.pop('data_load', None)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import GMEL
    from data_load import build_graph, get_scalers

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Helper: lazily yield (nfeat, adj, dis, od) per area ──────────────────
    def _iter_areas(areas):
        for area in areas:
            ap = os.path.join(data_path, area)
            nfeat = np.concatenate([
                np.load(os.path.join(ap, 'demos.npy')),
                np.load(os.path.join(ap, 'pois.npy')),
            ], axis=1)
            adj = np.load(os.path.join(ap, 'adj.npy'))
            dis = np.load(os.path.join(ap, 'dis.npy'))
            od  = np.load(os.path.join(ap, 'od.npy'))
            yield nfeat, adj, dis, od

    # ── Fit scalers on val data if not provided ───────────────────────────────
    if single_city_data is not None:
        train_mask = single_city_data['train_mask']
        toi = np.where(train_mask.any(1))[0]
        nf_fit = single_city_data['nfeat'][toi] if toi.size > 0 else single_city_data['nfeat']
        dis_fit = single_city_data['dis'][train_mask].reshape(-1, 1)
        if dis_fit.size == 0:
            dis_fit = single_city_data['dis'].reshape(-1, 1)
        od_fit = single_city_data['od_train'][train_mask].reshape(-1, 1)
        if od_fit.size == 0:
            od_fit = single_city_data['od_train'].reshape(-1, 1)
        # Preserve the semantic that scaled zero should inverse-transform back to zero.
        od_fit = np.concatenate([od_fit, np.zeros((1, 1), dtype=od_fit.dtype)], axis=0)
        nfeat_scaler = MinMaxScaler().fit(nf_fit)
        dis_scaler = MinMaxScaler().fit(dis_fit)
        od_scaler = MinMaxScaler().fit(od_fit)
    elif nfeat_scaler is None or dis_scaler is None or od_scaler is None:
        nf_v, di_v, od_v = [], [], []
        for nf, adj, dis, od in _iter_areas(train_areas):
            nf_v.append(nf); di_v.append(dis); od_v.append(od)
        nfeat_scaler, dis_scaler, od_scaler = get_scalers(nf_v, di_v, od_v)

    # ── Phase 1: Train GAT encoder ────────────────────────────────────────────
    gmel = GMEL().to(device)
    optimizer = torch.optim.Adam(gmel.parameters(), lr=encoder_lr)

    # Pre-load and cache data on GPU (avoid rebuilding graphs every epoch)
    # Full-matrix marginals for marginal loss (must NOT come from masked matrix)
    if single_city_data is not None:
        od_full_s = od_scaler.transform(
            single_city_data['od'].reshape(-1, 1)
        ).reshape(single_city_data['od'].shape)
        marginal_in_t = torch.FloatTensor(od_full_s.sum(0)).to(device)
        marginal_out_t = torch.FloatTensor(od_full_s.sum(1)).to(device)
    else:
        marginal_in_t = None
        marginal_out_t = None

    train_data_gpu = []
    if single_city_data is not None:
        nf = single_city_data['nfeat']
        adj = single_city_data['adj']
        dis = single_city_data['dis']
        od_train = single_city_data['od_train']
        train_mask = single_city_data['train_mask']
        nf_s = nfeat_scaler.transform(nf)
        od_s = _transform_masked_matrix(single_city_data['od'], od_scaler, train_mask)
        train_data_gpu.append((
            torch.FloatTensor(nf_s).to(device),
            build_graph(adj).to(device),
            torch.FloatTensor(od_s).to(device),
            torch.BoolTensor(train_mask).to(device),
            nf, adj, dis, od_train, train_mask.reshape(-1),
        ))
    else:
        for nf, adj, dis, od in _iter_areas(train_areas):
            nf_s = nfeat_scaler.transform(nf)
            od_s = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
            train_data_gpu.append((
                torch.FloatTensor(nf_s).to(device),
                build_graph(adj).to(device),
                torch.FloatTensor(od_s).to(device),
                None,
                nf, adj, dis, od, None,
            ))

    val_data_gpu = []
    if single_city_data is not None:
        nf = single_city_data['nfeat']
        adj = single_city_data['adj']
        dis = single_city_data['dis']
        od_val = single_city_data['od_val']
        val_mask = single_city_data['val_mask']
        nf_s = nfeat_scaler.transform(nf)
        od_s = _transform_masked_matrix(single_city_data['od'], od_scaler, val_mask)
        val_data_gpu.append((
            torch.FloatTensor(nf_s).to(device),
            build_graph(adj).to(device),
            torch.FloatTensor(od_s).to(device),
            torch.BoolTensor(val_mask).to(device),
            nf, adj, dis, od_val, val_mask.reshape(-1),
        ))
    else:
        for nf, adj, dis, od in _iter_areas(val_areas):
            nf_s = nfeat_scaler.transform(nf)
            od_s = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
            val_data_gpu.append((
                torch.FloatTensor(nf_s).to(device),
                build_graph(adj).to(device),
                torch.FloatTensor(od_s).to(device),
                None,
                nf, adj, dis, od, None,
            ))

    best_vl = np.inf
    best_pat = patience
    best_state = None
    train_losses = []
    val_losses = []
    pbar = tqdm(range(max_epochs), desc='GMEL-GAT', unit='ep', disable=not verbose)
    for ep in pbar:
        gmel.train()
        ep_losses = []
        for nf_t, g, od_t, od_mask_t, *_ in train_data_gpu:
            optimizer.zero_grad()
            flow_in, flow_out, flow, h_in, h_out = gmel(g, nf_t)
            # Use full-matrix marginals when available (single-city masked mode);
            # otherwise od_t is the full scaled matrix so .sum() is correct.
            m_in = marginal_in_t if marginal_in_t is not None else od_t.sum(0)
            m_out = marginal_out_t if marginal_out_t is not None else od_t.sum(1)
            loss = (_marginal_mse(flow_in, m_in) +
                    _marginal_mse(flow_out, m_out) +
                    _masked_mse(flow, od_t, od_mask_t))
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        gmel.eval()
        with torch.no_grad():
            vls = []
            for nf_t, g, od_t, od_mask_t, *_ in val_data_gpu:
                flow_in, flow_out, flow, _, _ = gmel(g, nf_t)
                m_in = marginal_in_t if marginal_in_t is not None else od_t.sum(0)
                m_out = marginal_out_t if marginal_out_t is not None else od_t.sum(1)
                vl = (_marginal_mse(flow_in, m_in) +
                      _marginal_mse(flow_out, m_out) +
                      _masked_mse(flow, od_t, od_mask_t)).item()
                vls.append(vl)
            vl = float(np.mean(vls))

        train_losses.append(float(np.mean(ep_losses)))
        val_losses.append(vl)
        pbar.set_postfix(loss=f'{train_losses[-1]:.4g}', val=f'{vl:.4g}', pat=best_pat)

        if vl < best_vl:
            best_vl = vl
            best_pat = patience
            best_state = {k: v.clone() for k, v in gmel.state_dict().items()}
        else:
            best_pat -= 1
            if best_pat == 0:
                break

    if best_state is not None:
        gmel.load_state_dict(best_state)

    saved_plot_path = save_loss_plot(
        train_losses,
        val_losses,
        title="GMEL Encoder Loss",
        save_path=loss_plot_path,
    )
    if saved_plot_path is not None and verbose:
        print(f"  -> Loss plot saved to {saved_plot_path}")
    gmel.train_losses = train_losses
    gmel.val_losses = val_losses
    gmel.loss_plot_path = str(saved_plot_path) if saved_plot_path is not None else None

    if verbose and single_city_data is not None:
        bilinear_pred = _predict_bilinear_matrix(gmel, val_data_gpu[0][0], val_data_gpu[0][1], od_scaler)
        _print_stage_metrics(
            "Bilinear Head",
            bilinear_pred,
            single_city_data['od'],
            single_city_data['test_mask'],
        )
    elif verbose:
        bilinear_metrics_full = []
        bilinear_metrics_nonzero = []
        for nf_t, g, _, _, _, _, _, od, _ in val_data_gpu:
            bilinear_pred = _predict_bilinear_matrix(gmel, nf_t, g, od_scaler)
            bilinear_metrics_full.append(cal_od_metrics(bilinear_pred, od))
            nz = od > 0
            if np.any(nz):
                bilinear_metrics_nonzero.append(compute_metrics(bilinear_pred[nz], od[nz]))
        _print_averaged_stage_metrics("Bilinear Head", bilinear_metrics_full, bilinear_metrics_nonzero)

    # ── Phase 2: Train GBRT on GAT embeddings ────────────────────────────────
    if verbose:
        print(f'  GMEL: fitting {decoder_type.upper()} on embeddings...')
    xtrain_emb, ytrain_emb = [], []
    xval_emb, yval_emb = [], []
    gmel.eval()
    with torch.no_grad():
        for nf_t, g, od_t, _, nf, adj, dis, od, fit_mask in train_data_gpu:
            _, _, _, h_in, h_out = gmel(g, nf_t)
            feat = _pair_embeddings_to_features(h_in.cpu().numpy(), h_out.cpu().numpy(), dis)
            y_flat = od.reshape(-1)
            if fit_mask is not None:
                xtrain_emb.append(feat[fit_mask])
                ytrain_emb.append(y_flat[fit_mask])
            else:
                xtrain_emb.append(feat)
                ytrain_emb.append(y_flat)
        for nf_t, g, od_t, _, nf, adj, dis, od, fit_mask in val_data_gpu:
            _, _, _, h_in, h_out = gmel(g, nf_t)
            feat = _pair_embeddings_to_features(h_in.cpu().numpy(), h_out.cpu().numpy(), dis)
            y_flat = od.reshape(-1)
            if fit_mask is not None:
                xval_emb.append(feat[fit_mask])
                yval_emb.append(y_flat[fit_mask])
            else:
                xval_emb.append(feat)
                yval_emb.append(y_flat)

    xtrain = np.concatenate(xtrain_emb)
    ytrain = np.concatenate(ytrain_emb)
    xval = np.concatenate(xval_emb) if xval_emb else None
    yval = np.concatenate(yval_emb) if yval_emb else None

    decoder = _fit_decoder(decoder_type, xtrain, ytrain, xval, yval, verbose=verbose, **decoder_kwargs)
    if verbose:
        print(f'  GMEL: {decoder_type.upper()} fitted.')

    if verbose and single_city_data is not None:
        nf_t, g, *_ = val_data_gpu[0]
        with torch.no_grad():
            _, _, _, h_in, h_out = gmel(g, nf_t)
        tree_pred = _predict_decoder_matrix(
            decoder,
            h_in.cpu().numpy(),
            h_out.cpu().numpy(),
            single_city_data['dis'],
        )
        _print_stage_metrics(
            "Tree Decoder",
            tree_pred,
            single_city_data['od'],
            single_city_data['test_mask'],
        )
    elif verbose:
        tree_metrics_full = []
        tree_metrics_nonzero = []
        with torch.no_grad():
            for nf_t, g, _, _, _, _, dis, od, _ in val_data_gpu:
                _, _, _, h_in, h_out = gmel(g, nf_t)
                tree_pred = _predict_decoder_matrix(
                    decoder,
                    h_in.cpu().numpy(),
                    h_out.cpu().numpy(),
                    dis,
                )
                tree_metrics_full.append(cal_od_metrics(tree_pred, od))
                nz = od > 0
                if np.any(nz):
                    tree_metrics_nonzero.append(compute_metrics(tree_pred[nz], od[nz]))
        _print_averaged_stage_metrics("Tree Decoder", tree_metrics_full, tree_metrics_nonzero)

    return gmel, decoder, nfeat_scaler, dis_scaler
