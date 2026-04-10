import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList, ReLU, Sequential

from torch_geometric.nn import GATConv, GINEConv, GPSConv

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



class GPSEncoder(nn.Module):
    def __init__(self, idim, hd, ped, ed, nl, nh=4, do=0.1, pe_type='rwpe', norm_type='batch_norm'):
        super().__init__()
        self.pe_type = pe_type
        self.use_pe = pe_type is not None
        self.norm_type = norm_type
        npd = hd - ped if self.use_pe else hd
        self.node_proj = Sequential(Linear(idim, npd), ReLU(), Linear(npd, npd))
        if self.use_pe:
            pid = PE_WALK_LEN
            if pe_type == 'spe':
                self.pe_abs_proj = Sequential(Linear(pid, ped), ReLU(), Linear(ped, ped))
                self.pe_norm = BatchNorm1d(pid)
                self.pe_proj = None
            else:
                self.pe_norm = BatchNorm1d(pid)
                self.pe_proj = Linear(pid, ped)
                self.pe_abs_proj = None
            if pe_type == 'rrwp':
                self.rrwp_proj = Sequential(Linear(PE_WALK_LEN, hd), ReLU(), Linear(hd, nh))
            else:
                self.rrwp_proj = None
        else:
            self.pe_abs_proj = None
            self.pe_norm = None
            self.pe_proj = None
            self.rrwp_proj = None
        self.edge_proj = Sequential(Linear(ed, hd), ReLU(), Linear(hd, hd))
        self.gps_layers = ModuleList()
        for _ in range(nl):
            gmlp = Sequential(Linear(hd, hd), ReLU(), Linear(hd, hd))
            gn = norm_type if norm_type == 'batch_norm' else None
            self.gps_layers.append(
                GPSConv(hd, GINEConv(gmlp), heads=nh, attn_type='multihead',
                        norm=gn, attn_kwargs={'dropout': do})
            )
        if norm_type == 'graph_norm':
            NC = GraphNormLayer
            self.extra_norms = ModuleList([NC(hd) for _ in range(nl)])
        else:
            self.extra_norms = None

    def forward(self, gd):
        ne = self.node_proj(gd.x)
        if self.use_pe:
            pr = gd.pe
            if self.pe_type == 'spe':
                pe = self.pe_abs_proj(torch.abs(self.pe_norm(pr)))
            else:
                pe = self.pe_proj(self.pe_norm(pr))
            h = torch.cat([ne, pe], dim=-1)
        else:
            h = ne
        ee = self.edge_proj(gd.edge_attr)
        batch = torch.zeros(gd.x.size(0), dtype=torch.long, device=gd.x.device)
        for i, l in enumerate(self.gps_layers):
            h = l(h, gd.edge_index, batch, edge_attr=ee)
            if self.extra_norms is not None:
                h = self.extra_norms[i](h, batch)
        return h


def _make_gat_conv(hd, nh, do, ed):
    if hd % nh != 0:
        raise ValueError(f"GAT hidden_dim={hd} must be divisible by heads={nh}")
    kwargs = dict(heads=nh, concat=True, dropout=do)
    if ed is not None and ed > 0:
        try:
            return GATConv(hd, hd // nh, edge_dim=ed, **kwargs), True
        except TypeError:
            pass
    return GATConv(hd, hd // nh, **kwargs), False


class GATEncoder(nn.Module):
    """GAT encoder used for the original-style GAT-GAN ablation."""
    def __init__(
        self,
        idim,
        hd,
        ped,
        ed,
        nl,
        nh=4,
        do=0.1,
        pe_type=None,
        norm_type='batch_norm',
        noise_dim=0,
    ):
        super().__init__()
        self.pe_type = pe_type
        self.use_pe = pe_type is not None
        self.norm_type = norm_type
        self.dropout = do
        self.noise_dim = noise_dim
        self.force_noise = False
        npd = hd - ped if self.use_pe else hd
        self.node_proj = Sequential(Linear(idim + noise_dim, npd), ReLU(), Linear(npd, npd))
        if self.use_pe:
            pid = PE_WALK_LEN
            self.pe_norm = BatchNorm1d(pid)
            self.pe_proj = Linear(pid, ped)
        else:
            self.pe_norm = None
            self.pe_proj = None

        self.gat_layers = ModuleList()
        self.gat_uses_edge_attr = []
        self.norms = ModuleList()
        for _ in range(nl):
            conv, use_edge_attr = _make_gat_conv(hd, nh, do, ed)
            self.gat_layers.append(conv)
            self.gat_uses_edge_attr.append(use_edge_attr)
            if norm_type == 'graph_norm':
                self.norms.append(GraphNormLayer(hd))
            elif norm_type == 'batch_norm':
                self.norms.append(BatchNorm1d(hd))
            else:
                self.norms.append(nn.Identity())

    def forward(self, gd):
        x = gd.x
        if self.noise_dim:
            if self.training or self.force_noise:
                noise = torch.randn(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype)
            else:
                noise = torch.zeros(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, noise], dim=-1)
        h = self.node_proj(x)
        if self.use_pe:
            pe = self.pe_proj(self.pe_norm(gd.pe))
            h = torch.cat([h, pe], dim=-1)

        batch = torch.zeros(gd.x.size(0), dtype=torch.long, device=gd.x.device)
        edge_attr = getattr(gd, 'edge_attr', None)
        for conv, use_edge_attr, norm in zip(self.gat_layers, self.gat_uses_edge_attr, self.norms):
            if use_edge_attr and edge_attr is not None:
                h = conv(h, gd.edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, gd.edge_index)
            h = F.elu(h)
            if isinstance(norm, GraphNormLayer):
                h = norm(h, batch)
            else:
                h = norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class BilinearDecoder(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hd, hd) * 0.01)

    def forward(self, oe, de, d=None, extra=None):
        return (oe * (de @ self.W.T)).sum(-1)


class LinearPairDecoder(nn.Module):
    def __init__(self, hd, extra_dim=0):
        super().__init__()
        self.net = Sequential(Linear(hd * 2 + 1 + extra_dim, hd), ReLU(), Linear(hd, 1))

    def forward(self, oe, de, d=None, extra=None):
        if d is None:
            d = torch.zeros(oe.size(0), 1, device=oe.device, dtype=oe.dtype)
        parts = [oe, de, d]
        if extra is not None:
            parts.append(extra)
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


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


class GravityGuidedDecoder(nn.Module):
    """ODGN-style gravity decoder.

    Implements the paper architecture: the node embedding is **split** into two
    equal halves — a mass part and a location part — matching the description:
        "we split the final node embedding into two parts. One part
        characterizes the mass of the region and the other part characterizes
        the location of the region in the abstract feature space."

    A single mass_head is shared for both origin and destination (the paper
    does not use separate role-specific mass heads).

    Returns log-gravity scores:
        log T_ij = log G + λ1·log M(i) + λ2·log M(j) − λ3·log(1 + r_ij)
    """
    def __init__(self, hd, loc_dim=None, extra_dim=0, eps=1e-6):
        super().__init__()
        # loc_dim kept for API compatibility but ignored — split is always hd//2
        self.mass_dim = hd // 2
        self.loc_dim_actual = hd - self.mass_dim
        self.eps = eps
        # Single shared mass head for both origin and destination (paper: one mass)
        self.mass_head = Sequential(
            Linear(self.mass_dim, self.mass_dim), ReLU(), Linear(self.mass_dim, 1)
        )
        self.loc_proj = Linear(self.loc_dim_actual, self.loc_dim_actual)
        self.log_g = nn.Parameter(torch.zeros(()))
        self.lambda_origin = nn.Parameter(torch.ones(()))
        self.lambda_dest = nn.Parameter(torch.ones(()))
        self.lambda_dist = nn.Parameter(torch.ones(()))
        self.geo_scale = nn.Parameter(torch.zeros(()))
        self.extra_bias = Linear(extra_dim, 1) if extra_dim else None
        if self.extra_bias is not None:
            nn.init.zeros_(self.extra_bias.weight)
            nn.init.zeros_(self.extra_bias.bias)

    def forward(self, oe, de, d=None, extra=None):
        # Split embedding: first half → mass, second half → location
        mo = F.softplus(self.mass_head(oe[..., :self.mass_dim])).squeeze(-1) + self.eps
        md = F.softplus(self.mass_head(de[..., :self.mass_dim])).squeeze(-1) + self.eps
        lo = self.loc_proj(oe[..., self.mass_dim:])
        ld = self.loc_proj(de[..., self.mass_dim:])
        dist = torch.linalg.vector_norm(lo - ld, dim=-1)
        if d is not None:
            # Distances are MinMax-scaled in data_load and may be negative off
            # the train range, so clamp before using them as a gravity radius.
            dist = dist + F.softplus(self.geo_scale) * d.squeeze(-1).clamp_min(0.0)
        dist = dist.clamp_min(self.eps)

        l1 = F.softplus(self.lambda_origin) + self.eps
        l2 = F.softplus(self.lambda_dest) + self.eps
        l3 = F.softplus(self.lambda_dist) + self.eps
        score = self.log_g + l1 * torch.log(mo) + l2 * torch.log(md) - l3 * torch.log1p(dist)
        if self.extra_bias is not None and extra is not None:
            score = score + self.extra_bias(extra).squeeze(-1)
        return score


def make_pair_decoder(dt, hd, th=4, tl=2, tdo=0.1, extra_dim=0):
    if dt == 'bilinear':
        return BilinearDecoder(hd)
    if dt == 'linear':
        return LinearPairDecoder(hd, extra_dim=extra_dim)
    if dt == 'transflower':
        return TransFlowerDecoder(hd, th, tl, tdo, extra_dim=extra_dim)
    if dt == 'gravity_guided':
        return GravityGuidedDecoder(hd, extra_dim=extra_dim)
    raise ValueError(dt)


class MLPEncoder(nn.Module):
    def __init__(self, idim, hd):
        super().__init__()
        self.net = Sequential(Linear(idim, hd), ReLU(), Linear(hd, hd))

    def forward(self, x):
        return self.net(x)


class GPSODModel(nn.Module):
    def __init__(self, idim, hd, ped, ed, gl, gh, gdo,
                 dt='bilinear', th=4, tl=2, tdo=0.1, pe_type='rwpe', nt='batch_norm', rle=None):
        super().__init__()
        self.encoder = GPSEncoder(idim, hd, ped, ed, gl, gh, gdo, pe_type=pe_type, norm_type=nt)
        self.decoder_type = dt
        self.hidden_dim = hd
        self.rle = rle
        rle_dim = rle.out_dim if rle else 0
        self.decoder = make_pair_decoder(dt, hd, th, tl, tdo, extra_dim=rle_dim)
        self.outflow_head = Linear(hd, 1)
        self.inflow_head = Linear(hd, 1)

    def encode(self, gd):
        return self.encoder(gd)

    def decode_row(self, ne, oi, di, dm, coords=None):
        D = di.size(0)
        oe = ne[oi].unsqueeze(0).expand(D, -1)
        de = ne[di]
        dist = dm[oi, di].unsqueeze(-1)
        extra = None
        if self.rle is not None:
            if coords is not None:
                rel = coords[oi].unsqueeze(0).expand(D, -1) - coords[di]
                extra = self.rle(rel)
            else:
                extra = torch.zeros(D, self.rle.out_dim, device=ne.device)
        return self.decoder(oe, de, dist, extra)

    def predict_node_flows(self, ne):
        return self.outflow_head(ne).squeeze(-1), self.inflow_head(ne).squeeze(-1)


class GATODModel(nn.Module):
    def __init__(self, idim, hd, ped, ed, gl, gh, gdo,
                 dt='linear', th=4, tl=2, tdo=0.1, pe_type=None, nt='batch_norm', rle=None,
                 noise_dim=0):
        super().__init__()
        self.encoder = GATEncoder(
            idim, hd, ped, ed, gl, gh, gdo,
            pe_type=pe_type, norm_type=nt, noise_dim=noise_dim,
        )
        self.decoder_type = dt
        self.hidden_dim = hd
        self.rle = rle
        rle_dim = rle.out_dim if rle else 0
        self.decoder = make_pair_decoder(dt, hd, th, tl, tdo, extra_dim=rle_dim)
        self.outflow_head = Linear(hd, 1)
        self.inflow_head = Linear(hd, 1)

    def encode(self, gd):
        return self.encoder(gd)

    def decode_row(self, ne, oi, di, dm, coords=None):
        D = di.size(0)
        oe = ne[oi].unsqueeze(0).expand(D, -1)
        de = ne[di]
        dist = dm[oi, di].unsqueeze(-1)
        extra = None
        if self.rle is not None:
            if coords is not None:
                rel = coords[oi].unsqueeze(0).expand(D, -1) - coords[di]
                extra = self.rle(rel)
            else:
                extra = torch.zeros(D, self.rle.out_dim, device=ne.device)
        return self.decoder(oe, de, dist, extra)

    def predict_node_flows(self, ne):
        return self.outflow_head(ne).squeeze(-1), self.inflow_head(ne).squeeze(-1)


class TransFlowerODModel(nn.Module):
    """MLP encoder (no graph) with a pair decoder and optional RLE."""
    def __init__(self, idim, hd, nh=4, nl=2, do=0.1, rle=None, decoder_type='transflower'):
        super().__init__()
        self.encoder = MLPEncoder(idim, hd)
        self.rle = rle
        self.decoder_type = decoder_type
        rle_dim = rle.out_dim if rle else 0
        self.decoder = make_pair_decoder(decoder_type, hd, nh, nl, do, extra_dim=rle_dim)
        self.outflow_head = Linear(hd, 1)
        self.inflow_head = Linear(hd, 1)

    def encode(self, gd):
        return self.encoder(gd.x)

    def decode_row(self, ne, oi, di, dm, coords=None):
        D = di.size(0)
        oe = ne[oi].unsqueeze(0).expand(D, -1)
        de = ne[di]
        dist = dm[oi, di].unsqueeze(-1)
        extra = None
        if self.rle is not None:
            if coords is not None:
                rel = coords[oi].unsqueeze(0).expand(D, -1) - coords[di]
                extra = self.rle(rel)
            else:
                extra = torch.zeros(D, self.rle.out_dim, device=ne.device)
        return self.decoder(oe, de, dist, extra)

    def predict_node_flows(self, ne):
        return self.outflow_head(ne).squeeze(-1), self.inflow_head(ne).squeeze(-1)


def make_model(config, input_dim=None, edge_dim=None, graph_data_ref=None):
    if input_dim is None and graph_data_ref is not None:
        input_dim = graph_data_ref.x.size(-1)
    if edge_dim is None and graph_data_ref is not None:
        edge_dim = graph_data_ref.edge_attr.size(-1)

    rle = None
    if config.use_rle:
        from .rle import RelativeLocationEncoder
        rle = RelativeLocationEncoder(
            freq=config.rle_freq, out_dim=config.rle_out_dim,
            lambda_min=config.rle_lambda_min, lambda_max=config.rle_lambda_max,
        )

    if config.encoder_type == 'mlp':
        assert input_dim is not None
        return TransFlowerODModel(
            input_dim, HIDDEN_DIM, TF_HEADS, TF_LAYERS, TF_DROPOUT,
            rle=rle, decoder_type=config.decoder_type,
        ).to(device)
    gnn_layers = config.gnn_layers or GPS_LAYERS
    gnn_heads = config.gnn_heads or GPS_HEADS
    if config.encoder_type == 'gat':
        assert input_dim and edge_dim
        return GATODModel(
            input_dim, HIDDEN_DIM, PE_DIM, edge_dim, gnn_layers, gnn_heads, GPS_DROPOUT,
            config.decoder_type, TF_HEADS, TF_LAYERS, TF_DROPOUT,
            config.pe_type, config.gps_norm_type, rle=rle, noise_dim=config.gan_noise_dim,
        ).to(device)
    else:  # 'gps'
        assert input_dim and edge_dim
        return GPSODModel(
            input_dim, HIDDEN_DIM, PE_DIM, edge_dim, gnn_layers, gnn_heads, GPS_DROPOUT,
            config.decoder_type, TF_HEADS, TF_LAYERS, TF_DROPOUT,
            config.pe_type, config.gps_norm_type, rle=rle,
        ).to(device)
