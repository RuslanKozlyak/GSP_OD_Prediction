import time
import gc

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from tqdm.auto import tqdm


def train(train_areas, val_areas, data_path,
          device=None, nfeat_scaler=None, dis_scaler=None, od_scaler=None,
          max_epochs=1000, patience=10):
    """Train GMEL (GAT encoder + GBRT decoder).

    Args:
        train_areas: list of area IDs for training
        val_areas:   list of area IDs for validation
        data_path:   path to data root directory
        device:      torch.device
        nfeat_scaler, dis_scaler, od_scaler: pre-fitted sklearn scalers
            (if None, fitted on val_areas data)
        max_epochs / patience: GAT training schedule

    Returns:
        (gmel_net, gbrt, nfeat_scaler, dis_scaler)
    """
    import os, sys
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
    if nfeat_scaler is None or dis_scaler is None or od_scaler is None:
        nf_v, di_v, od_v = [], [], []
        for nf, adj, dis, od in _iter_areas(val_areas):
            nf_v.append(nf); di_v.append(dis); od_v.append(od)
        nfeat_scaler, dis_scaler, od_scaler = get_scalers(nf_v, di_v, od_v)

    # ── Phase 1: Train GAT encoder ────────────────────────────────────────────
    gmel = GMEL().to(device)
    optimizer = torch.optim.Adam(gmel.parameters(), lr=3e-4)

    # Pre-load validation data (small set)
    val_data = list(_iter_areas(val_areas))

    best_vl = np.inf
    best_pat = patience
    pbar = tqdm(range(max_epochs), desc='GMEL-GAT', unit='ep')
    for ep in pbar:
        gmel.train()
        ep_losses = []
        for nf, adj, dis, od in _iter_areas(train_areas):
            nf_s  = nfeat_scaler.transform(nf)
            od_s  = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)

            nf_t  = torch.FloatTensor(nf_s).to(device)
            g     = build_graph(adj).to(device)
            od_t  = torch.FloatTensor(od_s).to(device)

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
            for nf, adj, dis, od in val_data:
                nf_s  = nfeat_scaler.transform(nf)
                od_s  = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
                nf_t  = torch.FloatTensor(nf_s).to(device)
                g     = build_graph(adj).to(device)
                od_t  = torch.FloatTensor(od_s).to(device)
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
        else:
            best_pat -= 1
            if best_pat == 0:
                break

    # ── Phase 2: Train GBRT on GAT embeddings ────────────────────────────────
    print('  GMEL: fitting GBRT on embeddings...')
    gbrt = GradientBoostingRegressor(
        n_estimators=20, min_samples_split=2, min_samples_leaf=2, max_depth=None
    )
    xtrain_emb, ytrain_emb = [], []
    gmel.eval()
    with torch.no_grad():
        for nf, adj, dis, od in _iter_areas(train_areas):
            nf_s = nfeat_scaler.transform(nf)
            nf_t = torch.FloatTensor(nf_s).to(device)
            g    = build_graph(adj).to(device)
            _, _, _, h_in, h_out = gmel(g, nf_t)
            h = np.concatenate([h_in.cpu().numpy(), h_out.cpu().numpy()], axis=1)
            n = h.shape[0]
            h_o  = h.reshape([n, 1, h.shape[1]]).repeat(n, axis=1)
            h_d  = h.reshape([1, n, h.shape[1]]).repeat(n, axis=0)
            feat = np.concatenate(
                [h_o, h_d, dis.reshape([n, n, 1])], axis=2
            ).reshape([-1, h.shape[1] * 2 + 1])
            xtrain_emb.append(feat)
            ytrain_emb.append(od.reshape(-1))

    gbrt.fit(np.concatenate(xtrain_emb), np.concatenate(ytrain_emb))
    print('  GMEL: GBRT fitted.')

    return gmel, gbrt, nfeat_scaler, dis_scaler


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_load import load_data, get_scalers, build_graph, load_all_areas, split_train_valid_test
    from metrics import cal_od_metrics, average_listed_metrics
    from pprint import pprint

    print("\n  **Loading data...")
    (nf_tr, adj_tr, dis_tr, od_tr,
     nf_val, adj_val, dis_val, od_val,
     nf_te, adj_te, dis_te, od_te) = load_data()

    areas = load_all_areas(if_shuffle=True)
    train_areas, val_areas, test_areas = split_train_valid_test(areas)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

    nfeat_scaler, dis_scaler, od_scaler = get_scalers(nf_tr, dis_tr, od_tr)

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gmel, gbrt, nfeat_scaler, dis_scaler = train(
        train_areas, val_areas, data_path, device=device_,
        nfeat_scaler=nfeat_scaler, dis_scaler=dis_scaler, od_scaler=od_scaler,
    )

    print("\n  **Evaluating...")
    metrics_all = []
    for nf, adj, dis, od in zip(nf_te, adj_te, dis_te, od_te):
        nf_s = nfeat_scaler.transform(nf)
        nf_t = torch.FloatTensor(nf_s).to(device_)
        g    = build_graph(adj).to(device_)
        with torch.no_grad():
            _, _, _, h_in, h_out = gmel(g, nf_t)
            h = np.concatenate([h_in.cpu().numpy(), h_out.cpu().numpy()], axis=1)
            n = h.shape[0]
            h_o  = h.reshape([n, 1, h.shape[1]]).repeat(n, axis=1)
            h_d  = h.reshape([1, n, h.shape[1]]).repeat(n, axis=0)
            feat = np.concatenate(
                [h_o, h_d, dis.reshape([n, n, 1])], axis=2
            ).reshape([-1, h.shape[1] * 2 + 1])
            od_hat = gbrt.predict(feat).reshape(n, n)
            od_hat[od_hat < 0] = 0
        metrics_all.append(cal_od_metrics(od_hat, od))

    pprint(average_listed_metrics(metrics_all))
