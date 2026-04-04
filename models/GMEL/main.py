import time
import gc

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm


def train(train_areas, val_areas, data_path,
          device=None, nfeat_scaler=None, dis_scaler=None, od_scaler=None,
          max_epochs=1000, patience=10, single_city_data=None):
    """Train GMEL (PyG GAT encoder + GBRT decoder).

    Args:
        train_areas: list of area IDs for training
        val_areas:   list of area IDs for validation
        data_path:   path to data root directory
        device:      torch.device
        nfeat_scaler, dis_scaler, od_scaler: pre-fitted sklearn scalers
            (if None, fitted on train_areas data)
        max_epochs / patience: GAT training schedule
        single_city_data: optional dict from prepare_single_city_graph() for
            honest train/val masking inside one city

    Returns:
        (gmel_net, gbrt, nfeat_scaler, dis_scaler)
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
    optimizer = torch.optim.Adam(gmel.parameters(), lr=3e-4)

    # Pre-load and cache data on GPU (avoid rebuilding graphs every epoch)
    train_data_gpu = []
    if single_city_data is not None:
        nf = single_city_data['nfeat']
        adj = single_city_data['adj']
        dis = single_city_data['dis']
        od_train = single_city_data['od_train']
        train_mask_flat = single_city_data['train_mask'].reshape(-1)
        nf_s = nfeat_scaler.transform(nf)
        od_s = od_scaler.transform(od_train.reshape(-1, 1)).reshape(od_train.shape)
        train_data_gpu.append((
            torch.FloatTensor(nf_s).to(device),
            build_graph(adj).to(device),
            torch.FloatTensor(od_s).to(device),
            nf, adj, dis, od_train, train_mask_flat,
        ))
    else:
        for nf, adj, dis, od in _iter_areas(train_areas):
            nf_s = nfeat_scaler.transform(nf)
            od_s = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
            train_data_gpu.append((
                torch.FloatTensor(nf_s).to(device),
                build_graph(adj).to(device),
                torch.FloatTensor(od_s).to(device),
                nf, adj, dis, od, None,
            ))

    val_data_gpu = []
    if single_city_data is not None:
        nf = single_city_data['nfeat']
        adj = single_city_data['adj']
        dis = single_city_data['dis']
        od_val = single_city_data['od_val']
        nf_s = nfeat_scaler.transform(nf)
        od_s = od_scaler.transform(od_val.reshape(-1, 1)).reshape(od_val.shape)
        val_data_gpu.append((
            torch.FloatTensor(nf_s).to(device),
            build_graph(adj).to(device),
            torch.FloatTensor(od_s).to(device),
        ))
    else:
        for nf, adj, dis, od in _iter_areas(val_areas):
            nf_s = nfeat_scaler.transform(nf)
            od_s = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
            val_data_gpu.append((
                torch.FloatTensor(nf_s).to(device),
                build_graph(adj).to(device),
                torch.FloatTensor(od_s).to(device),
            ))

    best_vl = np.inf
    best_pat = patience
    best_state = None
    pbar = tqdm(range(max_epochs), desc='GMEL-GAT', unit='ep')
    for ep in pbar:
        gmel.train()
        ep_losses = []
        for nf_t, g, od_t, *_ in train_data_gpu:
            optimizer.zero_grad()
            flow_in, flow_out, flow, h_in, h_out = gmel(g, nf_t)
            loss = (torch.mean((flow_in  - od_t.sum(0)) ** 2) +
                    torch.mean((flow_out - od_t.sum(1)) ** 2) +
                    torch.mean((flow     - od_t)        ** 2))
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        gmel.eval()
        with torch.no_grad():
            vls = []
            for nf_t, g, od_t in val_data_gpu:
                flow_in, flow_out, flow, _, _ = gmel(g, nf_t)
                vl = (torch.mean((flow_in  - od_t.sum(0)) ** 2) +
                      torch.mean((flow_out - od_t.sum(1)) ** 2) +
                      torch.mean((flow     - od_t)        ** 2)).item()
                vls.append(vl)
            vl = float(np.mean(vls))

        pbar.set_postfix(loss=f'{np.mean(ep_losses):.4g}', val=f'{vl:.4g}', pat=best_pat)

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

    # ── Phase 2: Train GBRT on GAT embeddings ────────────────────────────────
    print('  GMEL: fitting GBRT on embeddings...')
    gbrt = GradientBoostingRegressor(
        n_estimators=20, min_samples_split=2, min_samples_leaf=2, max_depth=None
    )
    xtrain_emb, ytrain_emb = [], []
    gmel.eval()
    with torch.no_grad():
        for nf_t, g, od_t, nf, adj, dis, od, fit_mask in train_data_gpu:
            _, _, _, h_in, h_out = gmel(g, nf_t)
            h = np.concatenate([h_in.cpu().numpy(), h_out.cpu().numpy()], axis=1)
            n = h.shape[0]
            h_o  = h.reshape([n, 1, h.shape[1]]).repeat(n, axis=1)
            h_d  = h.reshape([1, n, h.shape[1]]).repeat(n, axis=0)
            feat = np.concatenate(
                [h_o, h_d, dis.reshape([n, n, 1])], axis=2
            ).reshape([-1, h.shape[1] * 2 + 1])
            y_flat = od.reshape(-1)
            if fit_mask is not None:
                xtrain_emb.append(feat[fit_mask])
                ytrain_emb.append(y_flat[fit_mask])
            else:
                xtrain_emb.append(feat)
                ytrain_emb.append(y_flat)

    gbrt.fit(np.concatenate(xtrain_emb), np.concatenate(ytrain_emb))
    print('  GMEL: GBRT fitted.')

    return gmel, gbrt, nfeat_scaler, dis_scaler


if __name__ == '__main__':
    from pprint import pprint
    from models.shared.metrics import cal_od_metrics, average_listed_metrics
    from models.shared.data_load import (
        load_graph_data, get_scalers, build_pyg_graph,
        prepare_single_city_graph, split_multi_city_ids, SINGLE_CITY_ID, DEFAULT_DATA_PATH,
    )
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_load import build_graph

    data_path = str(DEFAULT_DATA_PATH)
    train_areas = [SINGLE_CITY_ID]
    val_areas = [SINGLE_CITY_ID]
    test_areas = [SINGLE_CITY_ID]
    single_city_data = prepare_single_city_graph(SINGLE_CITY_ID, data_path=data_path)

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gmel, gbrt, nfeat_scaler, dis_scaler = train(
        train_areas, val_areas, data_path, device=device_,
        single_city_data=single_city_data,
    )

    nf_te, adj_te, dis_te, od_te = load_graph_data(test_areas, data_path)

    print("\n  **Evaluating...")
    metrics_all = []
    for nf, adj, dis, od in zip(nf_te, adj_te, dis_te, od_te):
        nf_s = nfeat_scaler.transform(nf)
        nf_t = torch.FloatTensor(nf_s).to(device_)
        g = build_graph(adj).to(device_)
        with torch.no_grad():
            _, _, _, h_in, h_out = gmel(g, nf_t)
            h = np.concatenate([h_in.cpu().numpy(), h_out.cpu().numpy()], axis=1)
            n = h.shape[0]
            h_o = h.reshape([n, 1, h.shape[1]]).repeat(n, axis=1)
            h_d = h.reshape([1, n, h.shape[1]]).repeat(n, axis=0)
            feat = np.concatenate(
                [h_o, h_d, dis.reshape([n, n, 1])], axis=2
            ).reshape([-1, h.shape[1] * 2 + 1])
            od_hat = gbrt.predict(feat).reshape(n, n)
            od_hat[od_hat < 0] = 0
        metrics_all.append(cal_od_metrics(od_hat, od))

    pprint(average_listed_metrics(metrics_all))
