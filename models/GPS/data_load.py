import os

import numpy as np
import geopandas as gpd
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde

from .config import (
    DATA_PATH, SHP_PATH, PE_WALK_LEN, HUBER_KDE_BW, HUBER_MIN_PROB,
    N_DEST_SAMPLE, device, split_configured_multi_city_ids,
)
from .features import build_feature_matrix


def precompute_coords(data_path=DATA_PATH, shp_path=SHP_PATH):
    for aid in os.listdir(data_path):
        ad = os.path.join(data_path, aid)
        if not os.path.isdir(ad):
            continue
        cp = os.path.join(ad, "coords.npy")
        if os.path.exists(cp):
            continue
        sf = os.path.join(shp_path, aid, f"{aid}.shp")
        if not os.path.exists(sf):
            continue
        try:
            gdf = gpd.read_file(sf).to_crs("EPSG:3857")
            c = gdf.geometry.centroid
            np.save(cp, np.column_stack([c.x.values, c.y.values]))
        except Exception:
            pass


def load_area(area_id, data_path=DATA_PATH):
    ap = os.path.join(data_path, area_id)
    try:
        raw = {
            "demos": np.load(os.path.join(ap, "demos.npy")),
            "pois": np.load(os.path.join(ap, "pois.npy")),
            "lu": None,
            "jobs": None,
        }
        lp = os.path.join(ap, "lu.npy")
        jp = os.path.join(ap, "jobs.npy")
        if os.path.exists(lp):
            raw["lu"] = np.load(lp)
        if os.path.exists(jp):
            raw["jobs"] = np.load(jp)
        nf = build_feature_matrix(raw)
        adj = np.load(os.path.join(ap, "adj.npy"))
        dis = np.load(os.path.join(ap, "dis.npy"))
        od = np.load(os.path.join(ap, "od.npy"))
        cp = os.path.join(ap, "coords.npy")
        co = np.load(cp) if os.path.exists(cp) else None
        return nf, adj, dis, od, co
    except Exception:
        return None


def _compute_lape_pe(ds, out_dim):
    """Dual LaPE (WeDAN-style): distance-Laplacian + similarity-Laplacian eigenvectors.
    Returns (N, out_dim) array, with k = out_dim // 2 eigenvectors per Laplacian type.
    """
    k = out_dim // 2

    def _eig(A):
        D = np.diag(A.sum(1))
        _, V = np.linalg.eigh((D - A).astype(float))
        # eigh returns eigenvalues sorted ascending; columns of V are eigenvectors
        pe = V[:, :k].real
        if pe.shape[1] < k:
            pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
        return pe.astype(np.float32)

    sim_A = np.exp(-(ds ** 2) / (2.0 * 10000.0 ** 2))
    return np.concatenate([_eig(ds), _eig(sim_A)], axis=1)  # (N, 2k)


def build_graph(adjacency, nfs, ds, dev=None, pe_type='rwpe'):
    if dev is None:
        dev = device
    ri, ci = np.where(adjacency > 0)
    gd = Data(
        x=torch.FloatTensor(nfs),
        edge_index=torch.LongTensor(np.stack([ri, ci])),
        edge_attr=torch.FloatTensor(ds[ri, ci]).unsqueeze(-1),
        num_nodes=nfs.shape[0],
    )
    if pe_type is None:
        return gd.to(dev)
    if pe_type == 'rwpe':
        gd = T.AddRandomWalkPE(walk_length=PE_WALK_LEN, attr_name='pe')(gd)
    elif pe_type == 'spe':
        gd = T.AddLaplacianEigenvectorPE(k=PE_WALK_LEN, attr_name='pe')(gd)
    elif pe_type == 'rrwp':
        gd = T.AddRandomWalkPE(walk_length=PE_WALK_LEN, attr_name='pe')(gd)
        N = nfs.shape[0]
        a = torch.zeros(N, N)
        a[gd.edge_index[0], gd.edge_index[1]] = 1.0
        d = a.sum(1).clamp(min=1)
        rw = a / d.unsqueeze(1)
        feats = []
        rk = rw.clone()
        for k in range(PE_WALK_LEN):
            feats.append(rk[gd.edge_index[0], gd.edge_index[1]].unsqueeze(-1))
            if k < PE_WALK_LEN - 1:
                rk = rk @ rw
        gd.rrwp_edge = torch.cat(feats, dim=-1)
    elif pe_type == 'lape':
        gd.pe = torch.FloatTensor(_compute_lape_pe(ds, PE_WALK_LEN)).to(dev)
    else:
        raise ValueError(f"Unknown pe_type: {pe_type}")
    return gd.to(dev)


def build_huber_weight_table(od, fm, bw=HUBER_KDE_BW, mp=HUBER_MIN_PROB):
    fl = od[fm].ravel()
    fl = fl[fl > 0].astype(float)
    if len(fl) < 10:
        return None, None
    kde = gaussian_kde(fl, bw_method=bw / (fl.std() + 1e-8))
    fg = np.linspace(0, fl.max() * 1.05, 2000)
    return fg, 1.0 / np.maximum(kde(np.sqrt(np.maximum(fg, 0))), mp)


def interpolate_huber_weights(fv, fg, wt):
    if fg is None:
        return np.ones_like(fv, dtype=np.float32)
    w = np.interp(fv, fg, wt)
    w[fv == 0] = 1.0
    return w.astype(np.float32)


def build_dest_dict(od):
    d = {}
    for o, dest in zip(*np.where(od > 0)):
        d.setdefault(int(o), []).append(int(dest))
    return {k: np.array(v) for k, v in d.items()}


# ─── Single-city data preparation ────────────────────────────────────────────

def prepare_single_city_data(area_id=None, pe_type='rwpe', data_path=DATA_PATH, seed=42):
    from .config import SINGLE_CITY_ID
    if area_id is None:
        area_id = SINGLE_CITY_ID

    # Auto-generate coords.npy from shapefile if missing
    cp = os.path.join(data_path, area_id, "coords.npy")
    if not os.path.exists(cp):
        sf = os.path.join(SHP_PATH, area_id, f"{area_id}.shp")
        if os.path.exists(sf):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(sf).to_crs("EPSG:3857")
                c = gdf.geometry.centroid
                np.save(cp, np.column_stack([c.x.values, c.y.values]))
            except Exception:
                pass

    data = load_area(area_id, data_path)
    assert data is not None, f"Failed to load area {area_id}"
    node_features_raw, adjacency, distances, od_matrix_raw, coords = data
    num_nodes = node_features_raw.shape[0]

    np.random.seed(seed)
    nzo, nzd_ = np.where(od_matrix_raw > 0)
    np_ = len(nzo)
    perm = np.random.permutation(np_)
    nt = int(np_ * 0.8)
    nv = int(np_ * 0.9)

    train_mask = np.zeros((num_nodes, num_nodes), bool)
    val_mask = np.zeros((num_nodes, num_nodes), bool)
    test_mask = np.zeros((num_nodes, num_nodes), bool)
    train_mask[nzo[perm[:nt]], nzd_[perm[:nt]]] = True
    val_mask[nzo[perm[nt:nv]], nzd_[perm[nt:nv]]] = True
    test_mask[nzo[perm[nv:]], nzd_[perm[nv:]]] = True

    zo, zd = np.where(od_matrix_raw == 0)
    zero_rng = np.random.default_rng(seed + 1)
    zp = zero_rng.permutation(len(zo))
    zt = int(len(zo) * 0.8)
    zv = int(len(zo) * 0.9)
    train_full_mask = train_mask.copy()
    val_full_mask = val_mask.copy()
    test_full_mask = test_mask.copy()
    train_full_mask[zo[zp[:zt]], zd[zp[:zt]]] = True
    val_full_mask[zo[zp[zt:zv]], zd[zp[zt:zv]]] = True
    test_full_mask[zo[zp[zv:]], zd[zp[zv:]]] = True

    toi = np.where(train_mask.any(1))[0]
    node_features_scaled = MinMaxScaler().fit(node_features_raw[toi]).transform(node_features_raw)
    distances_scaled = MinMaxScaler().fit(distances[train_mask].reshape(-1, 1)).transform(
        distances.reshape(-1, 1)
    ).reshape(num_nodes, num_nodes)

    od_train = od_matrix_raw * train_mask
    od_val = od_matrix_raw * val_mask
    outflow_full = od_matrix_raw.sum(1)
    inflow_full = od_matrix_raw.sum(0)
    outflow_train = od_train.sum(1)
    inflow_train = od_train.sum(0)
    outflow_val = od_val.sum(1)
    inflow_val = od_val.sum(0)
    train_dest_dict = build_dest_dict(od_train)
    val_dest_dict = build_dest_dict(od_val)
    huber_flow_grid, huber_weight_table = build_huber_weight_table(od_matrix_raw, train_mask)

    gd = build_graph(adjacency, node_features_scaled, distances_scaled, device, pe_type=pe_type)
    dt = torch.FloatTensor(distances_scaled).to(device)

    coords_tensor = torch.FloatTensor(coords).to(device) if coords is not None else None

    return {
        'city_id': area_id, 'graph_data': gd, 'distance_matrix': dt,
        'od_matrix_np': od_matrix_raw, 'od_matrix_train': od_train,
        'outflow_full': outflow_full, 'inflow_full': inflow_full,
        'outflow_train': outflow_train, 'inflow_train': inflow_train,
        'outflow_val': outflow_val, 'inflow_val': inflow_val,
        'num_nodes': num_nodes, 'nonzero_dest_dict': train_dest_dict,
        'huber_flow_grid': huber_flow_grid, 'huber_weight_table': huber_weight_table,
        'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
        'train_full_mask': train_full_mask,
        'val_full_mask': val_full_mask,
        'test_full_mask': test_full_mask,
        'split_scope': 'single_city',
        'val_dest_dict': val_dest_dict,
        'node_features_scaled': node_features_scaled, 'distances_scaled': distances_scaled,
        'coords': coords, 'coords_tensor': coords_tensor,
    }


# ─── Multi-city data preparation ─────────────────────────────────────────────

def load_multi_city_raw(city_ids=None, data_path=DATA_PATH):
    from .config import MULTI_CITY_IDS
    if city_ids is None:
        city_ids = MULTI_CITY_IDS

    multi_city_raw = {}
    for cid in city_ids:
        data = load_area(cid, data_path)
        if data is None:
            print(f"  [SKIP] {cid}")
            continue
        nf, adj, dis, od, co = data
        if nf.shape[0] < 10:
            print(f"  [SKIP] {cid}: too few nodes")
            continue
        multi_city_raw[cid] = {'nfeat': nf, 'adj': adj, 'dis': dis, 'od': od, 'coords': co}
        print(f"  {cid}: N={nf.shape[0]}")

    common_feat_dim = min(v['nfeat'].shape[1] for v in multi_city_raw.values())
    for cid, raw in multi_city_raw.items():
        nf = raw['nfeat'][:, :common_feat_dim]
        dis = raw['dis']
        raw['nfeat_scaled'] = MinMaxScaler().fit(nf).transform(nf)
        raw['dis_scaled'] = MinMaxScaler().fit(
            dis.reshape(-1, 1)
        ).transform(dis.reshape(-1, 1)).reshape(nf.shape[0], nf.shape[0])

    return multi_city_raw


def split_multi_city(multi_city_raw, seed=42, val_size=2, test_size=2):
    mc_city_ids = list(multi_city_raw.keys())
    try:
        return split_configured_multi_city_ids(mc_city_ids)
    except ValueError:
        np.random.seed(seed)
        np.random.shuffle(mc_city_ids)
        n_total = len(mc_city_ids)
        if n_total <= val_size + test_size:
            raise ValueError(
                f"Need more than {val_size + test_size} cities for "
                f"train/val/test split, got {n_total}"
            )
        n_train = n_total - val_size - test_size
        train_city_ids = mc_city_ids[:n_train]
        val_city_ids = mc_city_ids[n_train:n_train + val_size]
        test_city_ids = mc_city_ids[n_train + val_size:]
        return mc_city_ids, train_city_ids, val_city_ids, test_city_ids


def prepare_city_data(cid, raw, pe_type='rwpe', data_path=DATA_PATH):
    # Auto-generate coords.npy from shapefile if missing
    cp = os.path.join(data_path, cid, "coords.npy")
    if not os.path.exists(cp):
        sf = os.path.join(SHP_PATH, cid, f"{cid}.shp")
        if os.path.exists(sf):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(sf).to_crs("EPSG:3857")
                c = gdf.geometry.centroid
                np.save(cp, np.column_stack([c.x.values, c.y.values]))
                raw['coords'] = np.load(cp)
            except Exception:
                pass

    nfs = raw['nfeat_scaled']
    ds = raw['dis_scaled']
    od = raw['od']
    nn_ = nfs.shape[0]
    g = build_graph(raw['adj'], nfs, ds, device, pe_type)
    dt = torch.FloatTensor(ds).to(device)
    nzd = build_dest_dict(od)
    hfg, hwt = build_huber_weight_table(od, od > 0)
    cnd = max(8, min(int(np.mean([len(v) for v in nzd.values()])), 128)) if nzd else N_DEST_SAMPLE
    coords = raw.get('coords')
    coords_tensor = torch.FloatTensor(coords).to(device) if coords is not None else None
    return {
        'city_id': cid, 'graph_data': g, 'distance_matrix': dt,
        'od_matrix_np': od, 'od_matrix_train': od,
        'outflow_full': od.sum(1), 'outflow_train': od.sum(1),
        'inflow_train': od.sum(0), 'inflow_full': od.sum(0),
        'outflow_val': od.sum(1), 'inflow_val': od.sum(0),
        'num_nodes': nn_, 'nonzero_dest_dict': nzd,
        'huber_flow_grid': hfg, 'huber_weight_table': hwt,
        'city_n_dest': cnd,
        'train_mask': od > 0, 'val_mask': od > 0, 'test_mask': od > 0,
        'split_scope': 'multi_city',
        'val_dest_dict': nzd,
        'node_features_scaled': nfs, 'distances_scaled': ds,
        'coords': coords, 'coords_tensor': coords_tensor,
    }


def prepare_multi_city_data(city_ids=None, pe_type='rwpe', data_path=DATA_PATH, seed=42):
    multi_city_raw = load_multi_city_raw(city_ids, data_path)
    mc_city_ids, train_city_ids, val_city_ids, test_city_ids = split_multi_city(multi_city_raw, seed)

    city_data_dict = {}
    for cid in mc_city_ids:
        city_data_dict[cid] = prepare_city_data(cid, multi_city_raw[cid], pe_type, data_path)
        print(f"  {cid}: N={city_data_dict[cid]['num_nodes']}")

    return city_data_dict, train_city_ids, val_city_ids, test_city_ids
