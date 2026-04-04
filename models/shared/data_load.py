"""Shared data loading utilities.

All models should use these functions for consistent data splits.
Supports two evaluation modes:
- Single city: one city, nonzero OD pairs split 80/10/10
- Multi city: 10 cities, split by city 8/1/1
"""
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data"

SINGLE_CITY_ID = "48201"
MULTI_CITY_IDS = [
    "17031", "48201", "04013", "06073", "06059",
    "36047", "12086", "48113", "06065", "36081",
]


# ─── Core loading ────────────────────────────────────────────────────────────

def load_area_raw(area_id, data_path=None):
    """Load raw numpy arrays for a single area."""
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    area_path = Path(data_path) / area_id
    return {
        "demos": np.load(area_path / "demos.npy"),
        "pois": np.load(area_path / "pois.npy"),
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


# ─── Flat feature construction ───────────────────────────────────────────────

def construct_flat_features(areas, data_path=None, feature_mode="full"):
    """Build (origin, dest, distance) feature matrices for a list of areas.

    Args:
        areas: list of area ID strings
        data_path: path to data directory
        feature_mode: "full" for demos+pois, "gravity" for demos[:,0] only

    Returns:
        xs: list of (N^2, F) arrays — one per area
        ys: list of (N^2,) arrays — OD flows flattened
    """
    xs, ys = [], []
    for area in areas:
        raw = load_area_raw(area, data_path)
        if feature_mode == "gravity":
            feat = raw["demos"][:, :1].astype(np.float32)
        else:
            feat = np.concatenate([raw["demos"], raw["pois"]], axis=1).astype(np.float32)
        dis = raw["dis"].astype(np.float32)
        n_nodes = feat.shape[0]
        feat_o = feat.reshape(n_nodes, 1, feat.shape[1]).repeat(n_nodes, axis=1)
        feat_d = feat.reshape(1, n_nodes, feat.shape[1]).repeat(n_nodes, axis=0)
        x = np.concatenate([feat_o, feat_d, dis.reshape(n_nodes, n_nodes, 1)], axis=2)
        xs.append(x.reshape(-1, feat.shape[1] * 2 + 1))
        ys.append(raw["od"].reshape(-1).astype(np.float32))
    return xs, ys


# ─── Single-city split (matches GPS prepare_single_city_data) ───────────────

def _make_single_city_masks(od_matrix, seed=42):
    """Create train/val/test masks on nonzero OD pairs (80/10/10 split).

    Uses the same seed and logic as models.GPS.data_load.prepare_single_city_data
    so that all models evaluate on identical splits.
    """
    num_nodes = od_matrix.shape[0]
    np.random.seed(seed)
    nzo, nzd = np.where(od_matrix > 0)
    np_ = len(nzo)
    perm = np.random.permutation(np_)
    nt = int(np_ * 0.8)
    nv = int(np_ * 0.9)

    train_mask = np.zeros((num_nodes, num_nodes), bool)
    val_mask = np.zeros((num_nodes, num_nodes), bool)
    test_mask = np.zeros((num_nodes, num_nodes), bool)
    train_mask[nzo[perm[:nt]], nzd[perm[:nt]]] = True
    val_mask[nzo[perm[nt:nv]], nzd[perm[nt:nv]]] = True
    test_mask[nzo[perm[nv:]], nzd[perm[nv:]]] = True

    return train_mask, val_mask, test_mask


def prepare_single_city_flat(area_id=None, data_path=None, seed=42, feature_mode="full"):
    """Prepare flat features for a single city with proper train/val/test split.

    Returns:
        x_train, y_train: concatenated training pair features and targets
        xs_val, ys_val: list of validation arrays (per-area format, single element)
        xs_val_full, ys_val_full: full-matrix view for full-matrix monitoring
        xs_test, ys_test: list of test arrays (per-area format, single element)
        od_matrix: full OD matrix
        train_mask, val_mask, test_mask: boolean masks
    """
    if area_id is None:
        area_id = SINGLE_CITY_ID

    raw = load_area_raw(area_id, data_path)
    od = raw["od"]
    train_mask, val_mask, test_mask = _make_single_city_masks(od, seed)

    if feature_mode == "gravity":
        feat = raw["demos"][:, :1].astype(np.float32)
    else:
        feat = np.concatenate([raw["demos"], raw["pois"]], axis=1).astype(np.float32)
    dis = raw["dis"].astype(np.float32)
    n = feat.shape[0]

    feat_o = feat.reshape(n, 1, feat.shape[1]).repeat(n, axis=1)
    feat_d = feat.reshape(1, n, feat.shape[1]).repeat(n, axis=0)
    x_full = np.concatenate([feat_o, feat_d, dis.reshape(n, n, 1)], axis=2).reshape(-1, feat.shape[1] * 2 + 1)
    y_full = od.reshape(-1).astype(np.float32)

    train_flat = train_mask.reshape(-1)
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
        'n_nodes': n,
        'area_id': area_id,
    }


def prepare_single_city_graph(area_id=None, data_path=None, seed=42):
    """Prepare raw graph inputs for a single city with 80/10/10 pair masks."""
    if area_id is None:
        area_id = SINGLE_CITY_ID

    raw = load_area_raw(area_id, data_path)
    od = raw["od"].astype(np.float32)
    train_mask, val_mask, test_mask = _make_single_city_masks(od, seed)

    return {
        'area_id': area_id,
        'nfeat': np.concatenate([raw["demos"], raw["pois"]], axis=1).astype(np.float32),
        'adj': raw["adj"],
        'dis': raw["dis"].astype(np.float32),
        'od': od,
        'od_train': (od * train_mask).astype(np.float32),
        'od_val': (od * val_mask).astype(np.float32),
        'od_test': (od * test_mask).astype(np.float32),
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    }


# ─── Multi-city split ────────────────────────────────────────────────────────

def split_multi_city_ids(city_ids=None, seed=42):
    """Split city IDs into train/val/test (8/1/1)."""
    if city_ids is None:
        city_ids = list(MULTI_CITY_IDS)
    else:
        city_ids = list(city_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(city_ids)
    return city_ids[:8], city_ids[8:9], city_ids[9:]


# ─── Graph data helpers (for GMEL, NetGAN) ──────────────────────────────────

def load_graph_data(areas, data_path=None):
    """Load graph data for multiple areas."""
    nfeats, adjs, dises, ods = [], [], [], []
    for area in areas:
        raw = load_area_raw(area, data_path)
        nfeats.append(np.concatenate([raw["demos"], raw["pois"]], axis=1))
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


def iter_graph_areas(areas, data_path=None):
    """Yield (nfeat, adj, dis, od) for each area."""
    for area in areas:
        raw = load_area_raw(area, data_path)
        yield np.concatenate([raw["demos"], raw["pois"]], axis=1), raw["adj"], raw["dis"], raw["od"]
