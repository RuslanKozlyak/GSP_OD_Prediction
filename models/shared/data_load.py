"""Shared data loading utilities.

All models should use these functions for consistent data splits.
Supports two evaluation modes:
- Single city: one city with configurable pair split mode
- Multi city: configured city set split into fixed train/val/test groups
"""
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.GPS.config import (
    MULTI_CITY_IDS,
    SINGLE_CITY_ID,
    normalize_pair_split_mode,
    split_configured_multi_city_ids,
)
from models.GPS.features import build_feature_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_PATH_WITH_LU = PROJECT_ROOT / "data_lu" / "data"
DEFAULT_DATA_PATH = _DEFAULT_DATA_PATH_WITH_LU if _DEFAULT_DATA_PATH_WITH_LU.exists() else (PROJECT_ROOT / "data")


# ─── Core loading ────────────────────────────────────────────────────────────

def load_area_raw(area_id, data_path=None):
    """Load raw numpy arrays for a single area."""
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    area_path = Path(data_path) / area_id
    lu_path = area_path / "lu.npy"
    jobs_path = area_path / "jobs.npy"
    return {
        "demos": np.load(area_path / "demos.npy"),
        "pois": np.load(area_path / "pois.npy"),
        "lu": np.load(lu_path) if lu_path.exists() else None,
        "jobs": np.load(jobs_path) if jobs_path.exists() else None,
        "adj": np.load(area_path / "adj.npy"),
        "dis": np.load(area_path / "dis.npy"),
        "od": np.load(area_path / "od.npy"),
    }


def get_all_areas(data_path=None):
    """List all area IDs in the data directory."""
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    data_root = Path(data_path)
    return sorted(p.name for p in data_root.iterdir() if p.is_dir())


def _select_node_features(raw, feature_mode="full"):
    if feature_mode == "gravity":
        return raw["demos"][:, :1].astype(np.float32)
    if feature_mode == "full":
        return build_feature_matrix(raw, feature_preset="all")
    if feature_mode == "reduced":
        return build_feature_matrix(raw, feature_preset="reduced")
    raise ValueError(
        f"Unknown feature_mode={feature_mode!r}. "
        "Valid options: 'full', 'reduced', 'gravity'"
    )


# ─── Flat feature construction ───────────────────────────────────────────────

def construct_flat_features(areas, data_path=None, feature_mode="full"):
    """Build (origin, dest, distance) feature matrices for a list of areas.

    Args:
        areas: list of area ID strings
        data_path: path to data directory
        feature_mode: "full" for all configured node features,
            "reduced" for the previous demo subset,
            "gravity" for demos[:,0] only

    Returns:
        xs: list of (N^2, F) arrays — one per area
        ys: list of (N^2,) arrays — OD flows flattened
    """
    xs, ys = [], []
    for area in areas:
        raw = load_area_raw(area, data_path)
        feat = _select_node_features(raw, feature_mode)
        dis = raw["dis"].astype(np.float32)
        n_nodes = feat.shape[0]
        feat_o = feat.reshape(n_nodes, 1, feat.shape[1]).repeat(n_nodes, axis=1)
        feat_d = feat.reshape(1, n_nodes, feat.shape[1]).repeat(n_nodes, axis=0)
        x = np.concatenate([feat_o, feat_d, dis.reshape(n_nodes, n_nodes, 1)], axis=2)
        xs.append(x.reshape(-1, feat.shape[1] * 2 + 1))
        ys.append(raw["od"].reshape(-1).astype(np.float32))
    return xs, ys


# ─── Single-city split (matches GPS prepare_single_city_data) ───────────────

def _split_mask_from_pairs(num_nodes, row_idx, col_idx, perm):
    train_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    val_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    test_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    n_pairs = len(row_idx)
    n_train = int(n_pairs * 0.8)
    n_val = int(n_pairs * 0.9)
    train_sel = perm[:n_train]
    val_sel = perm[n_train:n_val]
    test_sel = perm[n_val:]
    train_mask[row_idx[train_sel], col_idx[train_sel]] = True
    val_mask[row_idx[val_sel], col_idx[val_sel]] = True
    test_mask[row_idx[test_sel], col_idx[test_sel]] = True
    return train_mask, val_mask, test_mask


def build_single_city_pair_masks(od_matrix, seed=42, pair_split_mode='nonzero_pairs'):
    """Create single-city train/val/test masks for both pair-split regimes."""
    pair_split_mode = normalize_pair_split_mode(pair_split_mode)
    num_nodes = od_matrix.shape[0]
    positive_mask = np.asarray(od_matrix > 0, dtype=bool)

    if pair_split_mode == 'all_pairs':
        all_rows, all_cols = np.indices((num_nodes, num_nodes))
        all_rows = all_rows.reshape(-1)
        all_cols = all_cols.reshape(-1)
        perm = np.random.RandomState(seed).permutation(all_rows.shape[0])
        train_full_mask, val_full_mask, test_full_mask = _split_mask_from_pairs(
            num_nodes, all_rows, all_cols, perm,
        )
        train_mask = train_full_mask & positive_mask
        val_mask = val_full_mask & positive_mask
        test_mask = test_full_mask & positive_mask
    else:
        nzo, nzd = np.where(positive_mask)
        perm = np.random.RandomState(seed).permutation(len(nzo))
        train_mask, val_mask, test_mask = _split_mask_from_pairs(
            num_nodes, nzo, nzd, perm,
        )

        zo, zd = np.where(~positive_mask)
        zero_perm = np.random.default_rng(seed + 1).permutation(len(zo))
        z_train, z_val, z_test = _split_mask_from_pairs(
            num_nodes, zo, zd, zero_perm,
        )
        train_full_mask = train_mask | z_train
        val_full_mask = val_mask | z_val
        test_full_mask = test_mask | z_test

    train_fit_mask = train_full_mask if pair_split_mode == 'all_pairs' else train_mask
    val_fit_mask = val_full_mask if pair_split_mode == 'all_pairs' else val_mask
    test_fit_mask = test_full_mask if pair_split_mode == 'all_pairs' else test_mask

    return {
        'pair_split_mode': pair_split_mode,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_full_mask': train_full_mask,
        'val_full_mask': val_full_mask,
        'test_full_mask': test_full_mask,
        'train_fit_mask': train_fit_mask,
        'val_fit_mask': val_fit_mask,
        'test_fit_mask': test_fit_mask,
    }


def _make_single_city_masks(od_matrix, seed=42, pair_split_mode='nonzero_pairs'):
    masks = build_single_city_pair_masks(od_matrix, seed=seed, pair_split_mode=pair_split_mode)
    return masks['train_mask'], masks['val_mask'], masks['test_mask']


def _make_single_city_full_masks(od_matrix, train_mask, val_mask, test_mask, seed=42,
                                 pair_split_mode='nonzero_pairs'):
    del train_mask, val_mask, test_mask
    masks = build_single_city_pair_masks(od_matrix, seed=seed, pair_split_mode=pair_split_mode)
    return masks['train_full_mask'], masks['val_full_mask'], masks['test_full_mask']


def prepare_single_city_flat(area_id=None, data_path=None, seed=42, feature_mode="full",
                             pair_split_mode='nonzero_pairs'):
    """Prepare flat features for a single city with proper train/val/test split.

    Returns:
        x_train, y_train: concatenated training pair features and targets
        xs_val, ys_val: list of validation arrays (per-area format, single element)
        xs_val_full, ys_val_full: full-matrix view for full-matrix monitoring
        xs_test, ys_test: list of test arrays (per-area format, single element)
        od_matrix: full OD matrix
        train_mask, val_mask, test_mask: boolean nonzero-pair masks
    """
    if area_id is None:
        area_id = SINGLE_CITY_ID

    raw = load_area_raw(area_id, data_path)
    od = raw["od"]
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

    feat = _select_node_features(raw, feature_mode)
    dis = raw["dis"].astype(np.float32)
    n = feat.shape[0]

    feat_o = feat.reshape(n, 1, feat.shape[1]).repeat(n, axis=1)
    feat_d = feat.reshape(1, n, feat.shape[1]).repeat(n, axis=0)
    x_full = np.concatenate([feat_o, feat_d, dis.reshape(n, n, 1)], axis=2).reshape(-1, feat.shape[1] * 2 + 1)
    y_full = od.reshape(-1).astype(np.float32)

    train_flat = train_fit_mask.reshape(-1)
    val_flat = val_mask.reshape(-1)
    test_flat = test_mask.reshape(-1)

    x_train = x_full[train_flat]
    y_train = y_full[train_flat]
    xs_val = [x_full[val_flat]]
    ys_val = [y_full[val_flat]]
    xs_val_full = [x_full]
    ys_val_full = [y_full]
    xs_test = [x_full]  # full matrix for evaluation
    ys_test = [y_full]

    return {
        'x_train': x_train, 'y_train': y_train,
        'xs_val': xs_val, 'ys_val': ys_val,
        'xs_val_full': xs_val_full, 'ys_val_full': ys_val_full,
        'xs_test': xs_test, 'ys_test': ys_test,
        'x_full': x_full, 'y_full': y_full,
        'od_matrix': od,
        'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
        'train_full_mask': train_full_mask,
        'val_full_mask': val_full_mask,
        'test_full_mask': test_full_mask,
        'train_fit_mask': train_fit_mask,
        'val_fit_mask': val_fit_mask,
        'test_fit_mask': test_fit_mask,
        'pair_split_mode': normalize_pair_split_mode(pair_split_mode),
        'n_nodes': n,
        'area_id': area_id,
    }


def prepare_single_city_graph(area_id=None, data_path=None, seed=42, feature_mode="full",
                              pair_split_mode='nonzero_pairs'):
    """Prepare raw graph inputs for a single city with configurable pair masks."""
    if area_id is None:
        area_id = SINGLE_CITY_ID

    raw = load_area_raw(area_id, data_path)
    od = raw["od"].astype(np.float32)
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

    return {
        'area_id': area_id,
        'nfeat': _select_node_features(raw, feature_mode),
        'adj': raw["adj"],
        'dis': raw["dis"].astype(np.float32),
        'od': od,
        'od_train': (od * train_fit_mask).astype(np.float32),
        'od_val': (od * val_fit_mask).astype(np.float32),
        'od_test': (od * test_fit_mask).astype(np.float32),
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_full_mask': train_full_mask,
        'val_full_mask': val_full_mask,
        'test_full_mask': test_full_mask,
        'train_fit_mask': train_fit_mask,
        'val_fit_mask': val_fit_mask,
        'test_fit_mask': test_fit_mask,
        'pair_split_mode': normalize_pair_split_mode(pair_split_mode),
    }


# ─── Multi-city split ────────────────────────────────────────────────────────

def split_multi_city_ids(city_ids=None, seed=42):
    """Split city IDs into train/val/test using the configured fixed hold-out cities."""
    if city_ids is None:
        city_ids = list(MULTI_CITY_IDS)
    else:
        city_ids = list(city_ids)
    try:
        _, train_city_ids, val_city_ids, test_city_ids = split_configured_multi_city_ids(city_ids)
        return train_city_ids, val_city_ids, test_city_ids
    except ValueError:
        rng = np.random.RandomState(seed)
        rng.shuffle(city_ids)
        n_total = len(city_ids)
        if n_total < 3:
            raise ValueError(f"Need at least 3 cities for train/val/test split, got {n_total}")
        return city_ids[:-2], city_ids[-2:-1], city_ids[-1:]


# ─── Graph data helpers (for GMEL, NetGAN) ──────────────────────────────────

def load_graph_data(areas, data_path=None, feature_mode="full"):
    """Load graph data for multiple areas."""
    nfeats, adjs, dises, ods = [], [], [], []
    for area in areas:
        raw = load_area_raw(area, data_path)
        nfeats.append(_select_node_features(raw, feature_mode))
        adjs.append(raw["adj"])
        dises.append(raw["dis"])
        ods.append(raw["od"])
    return nfeats, adjs, dises, ods


def get_scalers(nfeats, dises, ods):
    """Fit MinMaxScalers on graph data."""
    nfeat_scaler = MinMaxScaler().fit(np.concatenate(nfeats, axis=0))
    dis_scaler = MinMaxScaler().fit(np.concatenate([d.reshape(-1, 1) for d in dises], axis=0))
    od_scaler = MinMaxScaler().fit(np.concatenate([o.reshape(-1, 1) for o in ods], axis=0))
    return nfeat_scaler, dis_scaler, od_scaler


def build_dgl_graph(adj, dev):
    """Build a DGL graph from an adjacency matrix."""
    import dgl
    import torch
    dst, src = adj.nonzero()
    weights = adj[adj.nonzero()]
    graph = dgl.graph((src, dst), num_nodes=adj.shape[0])
    graph.edata["d"] = torch.tensor(np.asarray(weights, dtype=np.float32).reshape(-1, 1))
    return graph.to(dev)


def build_pyg_graph(adj, dev):
    """Build a PyG graph from an adjacency matrix."""
    import torch
    from torch_geometric.data import Data

    dst, src = adj.nonzero()
    weights = np.asarray(adj[adj.nonzero()], dtype=np.float32).reshape(-1, 1)
    graph = Data(
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(weights, dtype=torch.float32),
        num_nodes=adj.shape[0],
    )
    return graph.to(dev)


def iter_graph_areas(areas, data_path=None, feature_mode="full"):
    """Yield (nfeat, adj, dis, od) for each area."""
    for area in areas:
        raw = load_area_raw(area, data_path)
        yield _select_node_features(raw, feature_mode), raw["adj"], raw["dis"], raw["od"]
