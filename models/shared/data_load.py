"""Shared data loading utilities.

All models should use these functions for consistent data splits.
Supports two evaluation modes:
- Single city: one city, nonzero OD pairs split 80/10/10
- Multi city: configured city set split into fixed train/val/test groups
"""
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.GPS.config import MULTI_CITY_IDS, SINGLE_CITY_ID, split_configured_multi_city_ids
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


def _make_single_city_full_masks(od_matrix, train_mask, val_mask, test_mask, seed=42):
    """Build matching split masks over all OD pairs, including true zeros."""
    num_nodes = od_matrix.shape[0]
    rng = np.random.default_rng(seed + 1)
    zo, zd = np.where(od_matrix == 0)
    perm = rng.permutation(len(zo))
    nt = int(len(zo) * 0.8)
    nv = int(len(zo) * 0.9)

    train_full_mask = train_mask.copy()
    val_full_mask = val_mask.copy()
    test_full_mask = test_mask.copy()
    train_full_mask[zo[perm[:nt]], zd[perm[:nt]]] = True
    val_full_mask[zo[perm[nt:nv]], zd[perm[nt:nv]]] = True
    test_full_mask[zo[perm[nv:]], zd[perm[nv:]]] = True

    assert train_full_mask.shape == (num_nodes, num_nodes)
    return train_full_mask, val_full_mask, test_full_mask


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
    train_full_mask, val_full_mask, test_full_mask = _make_single_city_full_masks(
        od, train_mask, val_mask, test_mask, seed
    )

    feat = _select_node_features(raw, feature_mode)
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
        'train_full_mask': train_full_mask,
        'val_full_mask': val_full_mask,
        'test_full_mask': test_full_mask,
        'n_nodes': n,
        'area_id': area_id,
    }


def prepare_single_city_graph(area_id=None, data_path=None, seed=42, feature_mode="full"):
    """Prepare raw graph inputs for a single city with 80/10/10 pair masks."""
    if area_id is None:
        area_id = SINGLE_CITY_ID

    raw = load_area_raw(area_id, data_path)
    od = raw["od"].astype(np.float32)
    train_mask, val_mask, test_mask = _make_single_city_masks(od, seed)
    train_full_mask, val_full_mask, test_full_mask = _make_single_city_full_masks(
        od, train_mask, val_mask, test_mask, seed
    )

    return {
        'area_id': area_id,
        'nfeat': _select_node_features(raw, feature_mode),
        'adj': raw["adj"],
        'dis': raw["dis"].astype(np.float32),
        'od': od,
        'od_train': (od * train_mask).astype(np.float32),
        'od_val': (od * val_mask).astype(np.float32),
        'od_test': (od * test_mask).astype(np.float32),
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_full_mask': train_full_mask,
        'val_full_mask': val_full_mask,
        'test_full_mask': test_full_mask,
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
