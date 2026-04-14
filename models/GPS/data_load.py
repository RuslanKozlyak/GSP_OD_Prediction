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
    MULTI_CITY_GLOBAL_SCALING, MULTI_CITY_FEATURE_PRUNE_MAX_ZERO_SHARE,
    MULTI_CITY_FEATURE_PRUNE_REMOVE_ZERO_VAR,
)
from .features import build_feature_matrix, get_feature_names_for_raw
from models.shared.data_load import build_single_city_pair_masks


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
        feature_names = get_feature_names_for_raw(raw)
        adj = np.load(os.path.join(ap, "adj.npy"))
        dis = np.load(os.path.join(ap, "dis.npy"))
        od = np.load(os.path.join(ap, "od.npy"))
        cp = os.path.join(ap, "coords.npy")
        co = np.load(cp) if os.path.exists(cp) else None
        return nf, adj, dis, od, co, feature_names
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
    return fg, 1.0 / np.maximum(kde(fg), mp)


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

def prepare_single_city_data(area_id=None, pe_type='rwpe', data_path=DATA_PATH, seed=42,
                             pair_split_mode='nonzero_pairs'):
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
    node_features_raw, adjacency, distances, od_matrix_raw, coords, feature_names_raw = data
    num_nodes = node_features_raw.shape[0]

    masks = build_single_city_pair_masks(
        od_matrix_raw, seed=seed, pair_split_mode=pair_split_mode,
    )
    train_mask = masks['train_mask']
    val_mask = masks['val_mask']
    test_mask = masks['test_mask']
    train_full_mask = masks['train_full_mask']
    val_full_mask = masks['val_full_mask']
    test_full_mask = masks['test_full_mask']
    train_fit_mask = masks['train_fit_mask']
    val_fit_mask = masks['val_fit_mask']
    test_fit_mask = masks['test_fit_mask']

    toi = np.where(train_fit_mask.any(1))[0]
    if toi.size == 0:
        toi = np.arange(num_nodes)
    node_features_scaled = MinMaxScaler().fit(node_features_raw[toi]).transform(node_features_raw)
    dist_fit = distances[train_fit_mask].reshape(-1, 1)
    if dist_fit.size == 0:
        dist_fit = distances.reshape(-1, 1)
    distances_scaled = MinMaxScaler().fit(dist_fit).transform(
        distances.reshape(-1, 1)
    ).reshape(num_nodes, num_nodes)

    od_train = od_matrix_raw * train_fit_mask
    od_val = od_matrix_raw * val_fit_mask
    od_test = od_matrix_raw * test_fit_mask
    outflow_full = od_matrix_raw.sum(1)
    inflow_full = od_matrix_raw.sum(0)
    outflow_train = od_train.sum(1)
    inflow_train = od_train.sum(0)
    outflow_val = od_val.sum(1)
    inflow_val = od_val.sum(0)
    train_dest_dict = build_dest_dict(od_train)
    val_dest_dict = build_dest_dict(od_val)
    huber_flow_grid, huber_weight_table = build_huber_weight_table(od_matrix_raw, train_fit_mask)

    gd = build_graph(adjacency, node_features_scaled, distances_scaled, device, pe_type=pe_type)
    dt = torch.FloatTensor(distances_scaled).to(device)

    coords_tensor = torch.FloatTensor(coords).to(device) if coords is not None else None

    return {
        'city_id': area_id, 'graph_data': gd, 'distance_matrix': dt,
        'od_matrix_np': od_matrix_raw, 'od_matrix_train': od_train,
        'od_matrix_val': od_val, 'od_matrix_test': od_test,
        'outflow_full': outflow_full, 'inflow_full': inflow_full,
        'outflow_train': outflow_train, 'inflow_train': inflow_train,
        'outflow_val': outflow_val, 'inflow_val': inflow_val,
        'num_nodes': num_nodes, 'nonzero_dest_dict': train_dest_dict,
        'huber_flow_grid': huber_flow_grid, 'huber_weight_table': huber_weight_table,
        'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
        'train_full_mask': train_full_mask,
        'val_full_mask': val_full_mask,
        'test_full_mask': test_full_mask,
        'train_fit_mask': train_fit_mask,
        'val_fit_mask': val_fit_mask,
        'test_fit_mask': test_fit_mask,
        'active_fit_mask': train_fit_mask,
        'active_origin_indices': np.where(train_fit_mask.any(1))[0],
        'val_origin_indices': np.where(val_fit_mask.any(1))[0],
        'test_origin_indices': np.where(test_fit_mask.any(1))[0],
        'pair_split_mode': masks['pair_split_mode'],
        'split_scope': 'single_city',
        'val_dest_dict': val_dest_dict,
        'node_features_scaled': node_features_scaled, 'distances_scaled': distances_scaled,
        'feature_names_raw': list(feature_names_raw),
        'feature_names_selected': list(feature_names_raw),
        'feature_keep_mask': np.ones(node_features_scaled.shape[1], dtype=bool),
        'feature_pruning_applied': False,
        'feature_scaling_scope': 'single_city_train_nodes',
        'distance_scaling_scope': 'single_city_train_pairs',
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
        nf, adj, dis, od, co, feature_names = data
        if nf.shape[0] < 10:
            print(f"  [SKIP] {cid}: too few nodes")
            continue
        multi_city_raw[cid] = {
            'nfeat': nf, 'adj': adj, 'dis': dis, 'od': od, 'coords': co,
            'feature_names_raw': list(feature_names),
        }
        print(f"  {cid}: N={nf.shape[0]}")
    return multi_city_raw


def _fit_multi_city_feature_pipeline(multi_city_raw, train_city_ids, city_seed_map, pair_split_mode):
    common_feat_dim = min(v['nfeat'].shape[1] for v in multi_city_raw.values())

    train_feature_blocks = []
    train_distance_blocks = []
    feature_names_raw = None
    for cid in train_city_ids:
        raw = multi_city_raw[cid]
        nf = raw['nfeat'][:, :common_feat_dim].astype(np.float32, copy=False)
        train_feature_blocks.append(nf)
        if feature_names_raw is None:
            feature_names_raw = list(raw.get('feature_names_raw', []))[:common_feat_dim]
        masks = build_single_city_pair_masks(
            raw['od'], seed=city_seed_map[cid], pair_split_mode=pair_split_mode,
        )
        dist_fit = raw['dis'][masks['train_fit_mask']].reshape(-1, 1)
        if dist_fit.size == 0:
            dist_fit = raw['dis'].reshape(-1, 1)
        train_distance_blocks.append(dist_fit.astype(np.float32, copy=False))

    train_features = np.concatenate(train_feature_blocks, axis=0)
    zero_share = np.mean(train_features == 0, axis=0)
    keep_mask = np.ones(train_features.shape[1], dtype=bool)
    if MULTI_CITY_FEATURE_PRUNE_REMOVE_ZERO_VAR:
        keep_mask &= np.nanstd(train_features, axis=0) > 0
    if MULTI_CITY_FEATURE_PRUNE_MAX_ZERO_SHARE is not None:
        keep_mask &= zero_share < float(MULTI_CITY_FEATURE_PRUNE_MAX_ZERO_SHARE)
    if not np.any(keep_mask):
        keep_mask = np.nanstd(train_features, axis=0) > 0
    if not np.any(keep_mask):
        keep_mask = np.ones(train_features.shape[1], dtype=bool)

    train_features = train_features[:, keep_mask]
    feature_scaler = MinMaxScaler().fit(train_features)
    train_distances = np.concatenate(train_distance_blocks, axis=0)
    distance_scaler = MinMaxScaler().fit(train_distances)

    selected_feature_names = None
    dropped_feature_names = None
    if feature_names_raw is not None:
        selected_feature_names = [
            name for name, keep in zip(feature_names_raw, keep_mask) if keep
        ]
        dropped_feature_names = [
            name for name, keep in zip(feature_names_raw, keep_mask) if not keep
        ]

    return {
        'common_feat_dim': common_feat_dim,
        'feature_keep_mask': keep_mask,
        'feature_scaler': feature_scaler,
        'distance_scaler': distance_scaler,
        'feature_zero_share_train': zero_share,
        'feature_names_raw': feature_names_raw,
        'feature_names_selected': selected_feature_names,
        'dropped_feature_names': dropped_feature_names,
    }


def _apply_multi_city_feature_pipeline(multi_city_raw, pipeline, use_global_scaling=True):
    common_feat_dim = pipeline['common_feat_dim']
    feature_keep_mask = pipeline['feature_keep_mask']
    feature_scaler = pipeline['feature_scaler'] if use_global_scaling else None
    distance_scaler = pipeline['distance_scaler'] if use_global_scaling else None
    feature_names_raw = pipeline.get('feature_names_raw')
    feature_names_selected = pipeline.get('feature_names_selected')
    dropped_feature_names = pipeline.get('dropped_feature_names')

    for cid, raw in multi_city_raw.items():
        nf = raw['nfeat'][:, :common_feat_dim].astype(np.float32, copy=False)
        nf = nf[:, feature_keep_mask]
        dis = raw['dis'].astype(np.float32, copy=False)

        if feature_scaler is not None:
            raw['nfeat_scaled'] = feature_scaler.transform(nf).astype(np.float32, copy=False)
            raw['feature_scaling_scope'] = 'global_train_cities'
        else:
            raw['nfeat_scaled'] = MinMaxScaler().fit(nf).transform(nf).astype(np.float32, copy=False)
            raw['feature_scaling_scope'] = 'per_city'

        if distance_scaler is not None:
            raw['dis_scaled'] = distance_scaler.transform(
                dis.reshape(-1, 1)
            ).reshape(dis.shape[0], dis.shape[1]).astype(np.float32, copy=False)
            raw['distance_scaling_scope'] = 'global_train_cities'
        else:
            raw['dis_scaled'] = MinMaxScaler().fit(
                dis.reshape(-1, 1)
            ).transform(dis.reshape(-1, 1)).reshape(dis.shape[0], dis.shape[1]).astype(np.float32, copy=False)
            raw['distance_scaling_scope'] = 'per_city'

        raw['feature_keep_mask'] = feature_keep_mask.copy()
        raw['feature_names_raw'] = list(feature_names_raw) if feature_names_raw is not None else None
        raw['feature_names_selected'] = (
            list(feature_names_selected) if feature_names_selected is not None else None
        )
        raw['dropped_feature_names'] = (
            list(dropped_feature_names) if dropped_feature_names is not None else None
        )
        raw['feature_pruning_applied'] = bool(np.any(~feature_keep_mask))
        raw['feature_dim_before_prune'] = int(common_feat_dim)
        raw['feature_dim_after_prune'] = int(np.sum(feature_keep_mask))
        raw['feature_zero_share_train'] = pipeline['feature_zero_share_train'].astype(np.float32, copy=False)


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


def prepare_city_data(cid, raw, pe_type='rwpe', data_path=DATA_PATH, seed=42,
                      pair_split_mode='nonzero_pairs'):
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
    masks = build_single_city_pair_masks(od, seed=seed, pair_split_mode=pair_split_mode)
    train_mask = masks['train_mask']
    val_mask = masks['val_mask']
    test_mask = masks['test_mask']
    train_full_mask = masks['train_full_mask']
    val_full_mask = masks['val_full_mask']
    test_full_mask = masks['test_full_mask']
    train_fit_mask = masks['train_fit_mask']
    val_fit_mask = masks['val_fit_mask']
    test_fit_mask = masks['test_fit_mask']
    od_train = od * train_fit_mask
    od_val = od * val_fit_mask
    od_test = od * test_fit_mask
    nzd = build_dest_dict(od_train)
    vdd = build_dest_dict(od_val)
    hfg, hwt = build_huber_weight_table(od, train_fit_mask)
    cnd = max(8, min(int(np.mean([len(v) for v in nzd.values()])), 128)) if nzd else N_DEST_SAMPLE
    coords = raw.get('coords')
    coords_tensor = torch.FloatTensor(coords).to(device) if coords is not None else None
    return {
        'city_id': cid, 'graph_data': g, 'distance_matrix': dt,
        'od_matrix_np': od, 'od_matrix_train': od_train,
        'od_matrix_val': od_val, 'od_matrix_test': od_test,
        'outflow_full': od.sum(1), 'outflow_train': od_train.sum(1),
        'inflow_train': od_train.sum(0), 'inflow_full': od.sum(0),
        'outflow_val': od_val.sum(1), 'inflow_val': od_val.sum(0),
        'num_nodes': nn_, 'nonzero_dest_dict': nzd,
        'huber_flow_grid': hfg, 'huber_weight_table': hwt,
        'city_n_dest': cnd,
        'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
        'train_full_mask': train_full_mask,
        'val_full_mask': val_full_mask,
        'test_full_mask': test_full_mask,
        'train_fit_mask': train_fit_mask,
        'val_fit_mask': val_fit_mask,
        'test_fit_mask': test_fit_mask,
        'active_fit_mask': train_fit_mask,
        'active_origin_indices': np.where(train_fit_mask.any(1))[0],
        'val_origin_indices': np.where(val_fit_mask.any(1))[0],
        'test_origin_indices': np.where(test_fit_mask.any(1))[0],
        'pair_split_mode': masks['pair_split_mode'],
        'split_scope': 'multi_city',
        'val_dest_dict': vdd,
        'node_features_scaled': nfs, 'distances_scaled': ds,
        'feature_names_raw': raw.get('feature_names_raw'),
        'feature_names_selected': raw.get('feature_names_selected'),
        'feature_keep_mask': raw.get('feature_keep_mask'),
        'feature_pruning_applied': raw.get('feature_pruning_applied', False),
        'feature_dim_before_prune': raw.get('feature_dim_before_prune', nfs.shape[1]),
        'feature_dim_after_prune': raw.get('feature_dim_after_prune', nfs.shape[1]),
        'feature_scaling_scope': raw.get('feature_scaling_scope', 'per_city'),
        'distance_scaling_scope': raw.get('distance_scaling_scope', 'per_city'),
        'dropped_feature_names': raw.get('dropped_feature_names'),
        'coords': coords, 'coords_tensor': coords_tensor,
    }


def prepare_multi_city_data(city_ids=None, pe_type='rwpe', data_path=DATA_PATH, seed=42,
                            pair_split_mode='nonzero_pairs'):
    multi_city_raw = load_multi_city_raw(city_ids, data_path)
    mc_city_ids, train_city_ids, val_city_ids, test_city_ids = split_multi_city(multi_city_raw, seed)
    city_seed_map = {cid: seed + idx for idx, cid in enumerate(mc_city_ids)}
    pipeline = _fit_multi_city_feature_pipeline(
        multi_city_raw, train_city_ids, city_seed_map, pair_split_mode,
    )
    _apply_multi_city_feature_pipeline(
        multi_city_raw, pipeline, use_global_scaling=MULTI_CITY_GLOBAL_SCALING,
    )
    kept = int(np.sum(pipeline['feature_keep_mask']))
    total = int(pipeline['common_feat_dim'])
    scaling_scope = 'global train-city scalers' if MULTI_CITY_GLOBAL_SCALING else 'per-city scalers'
    print(
        f"  Multi-city feature pipeline: {scaling_scope}, "
        f"kept {kept}/{total} features fit on train cities {train_city_ids}"
    )
    dropped = pipeline.get('dropped_feature_names') or []
    if dropped:
        preview = ", ".join(dropped[:8])
        suffix = " ..." if len(dropped) > 8 else ""
        print(f"  Pruned sparse/constant features: {preview}{suffix}")

    city_data_dict = {}
    for idx, cid in enumerate(mc_city_ids):
        city_data_dict[cid] = prepare_city_data(
            cid,
            multi_city_raw[cid],
            pe_type,
            data_path,
            seed=seed + idx,
            pair_split_mode=pair_split_mode,
        )
        print(f"  {cid}: N={city_data_dict[cid]['num_nodes']}")

    return city_data_dict, train_city_ids, val_city_ids, test_city_ids
