import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList, ReLU, Sequential

from torch_geometric.nn import GINEConv, GPSConv

from .config import (
    HIDDEN_DIM, PE_DIM, PE_WALK_LEN, GPS_HEADS, GPS_LAYERS, GPS_DROPOUT,
    TF_HEADS, TF_LAYERS, TF_DROPOUT, device,
)


class GraphNormLayer(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hd))
        self.beta = nn.Parameter(torch.zeros(hd))
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        mn = torch.zeros(batch.max() + 1, x.size(1), device=x.device)
        ct = torch.zeros(batch.max() + 1, 1, device=x.device)
        mn.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        ct.scatter_add_(0, batch.unsqueeze(1)[:, :1], torch.ones(x.size(0), 1, device=x.device))
        ct = ct.clamp(min=1)
        mn = mn / ct
        xs = x - self.alpha * mn[batch]
        v = torch.zeros(batch.max() + 1, x.size(1), device=x.device)
        v.scatter_add_(0, batch.unsqueeze(1).expand_as(xs), xs ** 2)
        v = v / ct
        return self.gamma * xs / (v[batch].sqrt() + 1e-5) + self.beta


class GRANOLANorm(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.bn = nn.LayerNorm(hd)
        self.pg = Sequential(Linear(hd, hd), ReLU(), Linear(hd, hd * 2))

    def forward(self, x, batch=None):
        xn = self.bn(x)
        p = self.pg(x.detach())
        g, b = p.chunk(2, dim=-1)
        return (1.0 + g) * xn + b


class GPSEncoder(nn.Module):
    def __init__(self, idim, hd, ped, ed, nl, nh=4, do=0.1, pe_type='rwpe', norm_type='batch_norm'):
        super().__init__()
        self.pe_type = pe_type
        self.norm_type = norm_type
        npd = hd - ped
        self.node_proj = Sequential(Linear(idim, npd), ReLU(), Linear(npd, npd))
        pid = PE_WALK_LEN
        if pe_type == 'spe':
            self.pe_abs_proj = Sequential(Linear(pid, ped), ReLU(), Linear(ped, ped))
            self.pe_norm = BatchNorm1d(pid)
        else:
            self.pe_norm = BatchNorm1d(pid)
            self.pe_proj = Linear(pid, ped)
        if pe_type == 'rrwp':
            self.rrwp_proj = Sequential(Linear(PE_WALK_LEN, hd), ReLU(), Linear(hd, nh))
        self.edge_proj = Sequential(Linear(ed, hd), ReLU(), Linear(hd, hd))
        self.gps_layers = ModuleList()
        for _ in range(nl):
            gmlp = Sequential(Linear(hd, hd), ReLU(), Linear(hd, hd))
            gn = norm_type if norm_type == 'batch_norm' else None
            self.gps_layers.append(
                GPSConv(hd, GINEConv(gmlp), heads=nh, attn_type='multihead',
                        norm=gn, attn_kwargs={'dropout': do})
            )
        if norm_type in ('graph_norm', 'granola'):
            NC = GraphNormLayer if norm_type == 'graph_norm' else GRANOLANorm
            self.extra_norms = ModuleList([NC(hd) for _ in range(nl)])
        else:
            self.extra_norms = None

    def forward(self, gd):
        ne = self.node_proj(gd.x)
        pr = gd.pe
        if self.pe_type == 'spe':
            pe = self.pe_abs_proj(torch.abs(self.pe_norm(pr)))
        else:
            pe = self.pe_proj(self.pe_norm(pr))
        h = torch.cat([ne, pe], dim=-1)
        ee = self.edge_proj(gd.edge_attr)
        batch = torch.zeros(gd.x.size(0), dtype=torch.long, device=gd.x.device)
        for i, l in enumerate(self.gps_layers):
            h = l(h, gd.edge_index, batch, edge_attr=ee)
            if self.extra_norms is not None:
                h = self.extra_norms[i](h, batch)
        return h


class BilinearDecoder(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hd, hd) * 0.01)

    def forward(self, oe, de, d=None):
        return (oe * (de @ self.W.T)).sum(-1)


class TransFlowerDecoder(nn.Module):
    def __init__(self, hd, nh=4, nl=2, do=0.1, extra_dim=0):
        super().__init__()
        self.fp = Sequential(Linear(hd * 2 + 1 + extra_dim, hd), ReLU(), Linear(hd, hd))
        tl = nn.TransformerEncoderLayer(
            d_model=hd, nhead=nh, dim_feedforward=hd * 4, dropout=do, batch_first=True
        )
        self.tf = nn.TransformerEncoder(tl, num_layers=nl)
        self.ph = Sequential(Linear(hd, hd // 2), ReLU(), Linear(hd // 2, 1))

    def forward(self, oe, de, d, extra=None):
        parts = [oe, de, d]
        if extra is not None:
            parts.append(extra)
        fe = self.fp(torch.cat(parts, dim=-1))
        return self.ph(self.tf(fe.unsqueeze(0)).squeeze(0)).squeeze(-1)


class GPSODModel(nn.Module):
    def __init__(self, idim, hd, ped, ed, gl, gh, gdo,
                 dt='bilinear', th=4, tl=2, tdo=0.1, pe_type='rwpe', nt='batch_norm'):
        super().__init__()
        self.encoder = GPSEncoder(idim, hd, ped, ed, gl, gh, gdo, pe_type=pe_type, norm_type=nt)
        self.decoder_type = dt
        self.hidden_dim = hd
        if dt == 'bilinear':
            self.decoder = BilinearDecoder(hd)
        elif dt == 'transflower':
            self.decoder = TransFlowerDecoder(hd, th, tl, tdo)
        else:
            raise ValueError(dt)
        self.outflow_head = Linear(hd, 1)
        self.inflow_head = Linear(hd, 1)

    def encode(self, gd):
        return self.encoder(gd)

    def decode_row(self, ne, oi, di, dm):
        D = di.size(0)
        return self.decoder(ne[oi].unsqueeze(0).expand(D, -1), ne[di], dm[oi, di].unsqueeze(-1))

    def predict_node_flows(self, ne):
        return self.outflow_head(ne).squeeze(-1), self.inflow_head(ne).squeeze(-1)


def make_model(config, input_dim=None, edge_dim=None, graph_data_ref=None):
    if input_dim is None and graph_data_ref is not None:
        input_dim = graph_data_ref.x.size(-1)
    if edge_dim is None and graph_data_ref is not None:
        edge_dim = graph_data_ref.edge_attr.size(-1)
    assert input_dim and edge_dim
    return GPSODModel(
        input_dim, HIDDEN_DIM, PE_DIM, edge_dim, GPS_LAYERS, GPS_HEADS, GPS_DROPOUT,
        config.decoder_type, TF_HEADS, TF_LAYERS, TF_DROPOUT, config.pe_type, config.gps_norm_type
    ).to(device)
