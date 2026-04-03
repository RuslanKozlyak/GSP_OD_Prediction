import math
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from .config import DATA_PATH, FLAT_CHUNK_SIZE, MULTI_CITY_IDS, SEED


def load_area_raw(area_id, data_path=DATA_PATH):
    area_path = Path(data_path) / area_id
    return {
        "demos": np.load(area_path / "demos.npy"),
        "pois": np.load(area_path / "pois.npy"),
        "adj": np.load(area_path / "adj.npy"),
        "dis": np.load(area_path / "dis.npy"),
        "od": np.load(area_path / "od.npy"),
    }



def get_all_areas(data_path=DATA_PATH):
    data_root = Path(data_path)
    return sorted(path.name for path in data_root.iterdir() if path.is_dir())



def split_areas(areas, seed=SEED):
    areas = list(areas)
    rng = np.random.RandomState(seed)
    rng.shuffle(areas)
    n_total = len(areas)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.9)
    return areas[:n_train], areas[n_train:n_val], areas[n_val:]



def split_multi_city_ids(city_ids=None, seed=SEED):
    ids = list(city_ids or MULTI_CITY_IDS)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    return ids[:8], ids[8:9], ids[9:]



def construct_flat_features(areas, data_path=DATA_PATH, feature_mode="full"):
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



def iter_flat_chunks(areas, data_path=DATA_PATH, feature_mode="full", chunk_size=FLAT_CHUNK_SIZE):
    for start in range(0, len(areas), chunk_size):
        chunk = areas[start:start + chunk_size]
        xs, ys = construct_flat_features(chunk, data_path, feature_mode)
        yield np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)



def count_chunks(areas, chunk_size=FLAT_CHUNK_SIZE):
    return math.ceil(len(areas) / chunk_size)



def load_graph_data(areas, data_path=DATA_PATH):
    nfeats, adjs, dises, ods = [], [], [], []
    for area in areas:
        raw = load_area_raw(area, data_path)
        nfeats.append(np.concatenate([raw["demos"], raw["pois"]], axis=1))
        adjs.append(raw["adj"])
        dises.append(raw["dis"])
        ods.append(raw["od"])
    return nfeats, adjs, dises, ods



def get_graph_scalers(nfeats, dises, ods):
    nfeat_scaler = MinMaxScaler().fit(np.concatenate(nfeats, axis=0))
    dis_scaler = MinMaxScaler().fit(np.concatenate([dis.reshape(-1, 1) for dis in dises], axis=0))
    od_scaler = MinMaxScaler().fit(np.concatenate([od.reshape(-1, 1) for od in ods], axis=0))
    return nfeat_scaler, dis_scaler, od_scaler



def iter_graph_areas(areas, data_path=DATA_PATH):
    for area in areas:
        raw = load_area_raw(area, data_path)
        yield np.concatenate([raw["demos"], raw["pois"]], axis=1), raw["adj"], raw["dis"], raw["od"]



def build_dgl_graph(adj, dev):
    import dgl

    dst, src = adj.nonzero()
    weights = adj[adj.nonzero()]
    graph = dgl.graph((src, dst), num_nodes=adj.shape[0])
    graph.edata["d"] = torch.tensor(np.asarray(weights, dtype=np.float32).reshape(-1, 1))
    return graph.to(dev)
