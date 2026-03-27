import numpy as np
import torch
import torch.nn.functional as F

from .config import HUBER_DELTA, LAMBDA_MAIN, LAMBDA_SUB, NORMALIZE_MULTITASK, device
from .data_load import interpolate_huber_weights


def sample_destinations(oi, nzd, nn, use_sampling=True, n_dest=128, inc_zeros=True, zr=0.3):
    nz = nzd.get(oi, np.array([], dtype=int))
    if not use_sampling:
        return np.arange(nn)
    if not inc_zeros:
        av = nz if len(nz) > 0 else np.arange(nn)
        return av if len(av) <= n_dest else np.random.choice(av, n_dest, replace=False)
    nzn = max(1, int(n_dest * zr))
    nnz = n_dest - nzn
    snz = nz if len(nz) <= nnz else np.random.choice(nz, nnz, replace=False)
    zd = np.setdiff1d(np.arange(nn), nz)
    sz = np.random.choice(zd, min(nzn, len(zd)), replace=False) if len(zd) > 0 else np.array([], dtype=int)
    return np.concatenate([snz, sz]).astype(int)


def compute_loss_for_city(model, cd, config, origin_batch_indices=None):
    lt = config.loss_type
    pm = config.prediction_mode
    ul = config.use_log_transform
    ne = model.encode(cd['graph_data'])
    nn_ = cd['num_nodes']
    od = cd['od_matrix_train']
    of = cd['outflow_train']
    inf_ = cd['inflow_train']
    nzd = cd['nonzero_dest_dict']
    ao = list(nzd.keys()) if origin_batch_indices is None else origin_batch_indices
    tl = torch.tensor(0.0, requires_grad=True, device=device)
    np_ = 0

    for oi in ao:
        di = sample_destinations(
            oi, nzd, nn_, config.use_dest_sampling,
            config.n_dest_sample, config.include_zero_pairs, config.zero_pair_ratio
        )
        if len(di) == 0:
            continue
        dt = torch.LongTensor(di).to(device)
        rf = od[oi, di].astype(float)
        tr = np.log1p(rf) if ul and lt != 'ce' else rf.copy()
        if pm == 'normalized':
            oof = of[oi]
            if oof < 1:
                continue
            tv = tr / (np.log1p(oof) + 1e-8) if ul and lt != 'ce' else tr / (oof + 1e-8)
        else:
            tv = tr
        tt = torch.FloatTensor(tv).to(device)
        sc = model.decode_row(ne, oi, dt, cd['distance_matrix'], coords=cd.get('coords_tensor'))

        if lt == 'huber':
            pr = F.softmax(sc, dim=0) if pm == 'normalized' else F.relu(sc)
            w = torch.FloatTensor(
                interpolate_huber_weights(rf, cd['huber_flow_grid'], cd['huber_weight_table'])
            ).to(device)
            df = torch.abs(pr - tt)
            h = torch.where(df <= HUBER_DELTA, 0.5 * df ** 2, HUBER_DELTA * df - 0.5 * HUBER_DELTA ** 2)
            rl = (w * h).mean()
        elif lt == 'ce':
            p = F.softmax(sc, dim=0)
            ct = torch.FloatTensor(rf / (of[oi] + 1e-8)).to(device)
            rl = -torch.sum(ct * torch.log(p + 1e-10))
        elif lt == 'focal':
            p = F.softmax(sc, dim=0)
            ct = torch.FloatTensor(rf / (of[oi] + 1e-8)).to(device)
            rl = -torch.sum(ct * (1.0 - p).pow(config.focal_gamma) * torch.log(p + 1e-10))
        elif lt == 'multitask':
            pr = F.softmax(sc, dim=0) if pm == 'normalized' else F.relu(sc)
            rl = F.mse_loss(pr, tt)
        elif lt == 'zinb':
            mu = F.softplus(sc) + 1e-4
            th = torch.ones_like(mu) * 10.0
            pi = torch.sigmoid(sc * 0.1)
            ft = torch.FloatTensor(rf).to(device)
            iz = (ft < 0.5).float()
            tc = th.clamp(min=1e-4)
            nb = (torch.lgamma(ft + tc) - torch.lgamma(tc) - torch.lgamma(ft + 1)
                  + tc * torch.log(tc / (tc + mu)) + ft * torch.log(mu / (tc + mu)))
            nz = torch.exp(tc * torch.log(tc / (tc + mu)))
            lp = iz * torch.log(pi + (1 - pi) * nz + 1e-10) + (1 - iz) * (torch.log(1 - pi + 1e-10) + nb)
            rl = -lp.mean()
        else:
            raise ValueError(lt)

        tl = tl + rl
        np_ += 1

    if np_ == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)
    ml = tl / np_

    if lt == 'multitask':
        po, pi_ = model.predict_node_flows(ne)
        if pm == 'normalized':
            tf_ = float(od.sum()) + 1e-8
            to_ = torch.FloatTensor(of / tf_).to(device)
            ti_ = torch.FloatTensor(inf_ / tf_).to(device)
        else:
            to_ = torch.FloatTensor(of).to(device)
            ti_ = torch.FloatTensor(inf_).to(device)
        ol = F.mse_loss(po, to_)
        il = F.mse_loss(pi_, ti_)
        if NORMALIZE_MULTITASK:
            nzf = od[od > 0].astype(float)
            tva = nzf / (od.sum() + 1e-8) if pm == 'normalized' else nzf
            dm = (torch.FloatTensor(tva).to(device) ** 2).mean() + 1e-8
            do_ = (to_ ** 2).mean() + 1e-8
            di_ = (ti_ ** 2).mean() + 1e-8
            ml = ml / dm
            ol = ol / do_
            il = il / di_
        return LAMBDA_MAIN * ml + (LAMBDA_SUB / 2) * (ol + il)
    return ml
