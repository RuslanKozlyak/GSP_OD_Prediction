import time
import os
import sys

import numpy as np
import torch

from models.shared.metrics import canonical_od_metrics
from models.shared.data_load import load_graph_data, get_scalers, build_dgl_graph


def _transform_masked_matrix(matrix, scaler, mask=None):
    if mask is None:
        return scaler.transform(matrix.reshape(-1, 1)).reshape(matrix.shape)
    scaled = np.zeros_like(matrix, dtype=np.float32)
    if mask.any():
        scaled[mask] = scaler.transform(matrix[mask].reshape(-1, 1)).reshape(-1)
    return scaled


def train(train_areas, valid_areas, data_path,
          device=None, nfeat_scaler=None, dis_scaler=None, od_scaler=None,
          n_epochs=2, lr=3e-4, gp_lambda=10, batch_size=128, verbose=1,
          single_city_data=None):
    """Train NetGAN (GAT + Generator + Discriminator).

    Args:
        train_areas: list of area IDs for training
        valid_areas: list of area IDs for validation
        data_path: path to data root
        device: torch device
        nfeat_scaler, dis_scaler, od_scaler: pre-fitted scalers
        n_epochs: number of GAN training epochs
        lr: generator/discriminator learning rate
        gp_lambda: gradient penalty weight
        batch_size: random-walk batch size
        verbose: enables training log messages

    Returns:
        dict with 'generator', 'nfeat_scaler', 'dis_scaler', 'od_scaler'
    """
    # Import model from local directory
    model_dir = os.path.dirname(os.path.abspath(__file__))
    old_path = sys.path.copy()
    sys.path.insert(0, model_dir)
    try:
        from model import Generator, Discriminator, sample_one_random_walk, sample_batch_real, compute_gradient_penalty
    finally:
        sys.path = old_path

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fit scalers on train data if not provided
    if single_city_data is not None:
        train_mask = single_city_data['train_mask']
        toi = np.where(train_mask.any(1))[0]
        nf_fit = single_city_data['nfeat'][toi] if toi.size > 0 else single_city_data['nfeat']
        dis_fit = single_city_data['dis'][train_mask].reshape(-1, 1)
        if dis_fit.size == 0:
            dis_fit = single_city_data['dis'].reshape(-1, 1)
        od_fit = single_city_data['od_train'][train_mask].reshape(-1, 1)
        if od_fit.size == 0:
            od_fit = single_city_data['od_train'].reshape(-1, 1)
        if nfeat_scaler is None:
            from sklearn.preprocessing import MinMaxScaler
            nfeat_scaler = MinMaxScaler().fit(nf_fit)
        if dis_scaler is None:
            from sklearn.preprocessing import MinMaxScaler
            dis_scaler = MinMaxScaler().fit(dis_fit)
        if od_scaler is None:
            from sklearn.preprocessing import MinMaxScaler
            od_scaler = MinMaxScaler().fit(od_fit)
        nf_tr = [single_city_data['nfeat']]
        adj_tr = [single_city_data['adj']]
        dis_tr = [single_city_data['dis']]
        od_tr = [single_city_data['od']]
        od_masks_tr = [train_mask]
    else:
        if nfeat_scaler is None or dis_scaler is None or od_scaler is None:
            nf_v, _, dis_v, od_v = load_graph_data(train_areas, data_path)
            nfeat_scaler, dis_scaler, od_scaler = get_scalers(nf_v, dis_v, od_v)
        nf_tr, adj_tr, dis_tr, od_tr = load_graph_data(train_areas, data_path)
        od_masks_tr = [None] * len(od_tr)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    if verbose:
        print(f'  NetGAN: training for {n_epochs} epochs...')
    t0 = time.time()
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        for nf, adj, dis, od, od_mask in zip(nf_tr, adj_tr, dis_tr, od_tr, od_masks_tr):
            nf_s = nfeat_scaler.transform(nf)
            dis_s = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)

            nf_t = torch.FloatTensor(nf_s).to(device)
            g = build_dgl_graph(adj, device)
            dis_t = torch.FloatTensor(dis_s).to(device)

            opt_g.zero_grad()
            fake_batch = generator.sample_generated_batch(g, nf_t, dis_t, batch_size).to(device)
            loss_g = -torch.mean(discriminator(fake_batch))
            loss_g.backward()
            opt_g.step()

            if epoch % 5 == 0:
                opt_d.zero_grad()
                with torch.no_grad():
                    _, adjacency, logp = generator.generate_OD_net(g, nf_t, dis_t)
                    batch = [sample_one_random_walk(adjacency, logp) for _ in range(batch_size)]
                    fake_batch = torch.stack(batch).to(device)

                od_s = _transform_masked_matrix(od, od_scaler, od_mask)
                real_batch = torch.FloatTensor(sample_batch_real(od_s)).to(device)
                loss_d = (torch.mean(discriminator(fake_batch))
                          - torch.mean(discriminator(real_batch))
                          + gp_lambda * compute_gradient_penalty(discriminator, real_batch, fake_batch))
                loss_d.backward()
                opt_d.step()

    if verbose:
        print(f'  NetGAN: trained in {time.time() - t0:.1f}s')

    return {
        'generator': generator,
        'nfeat_scaler': nfeat_scaler,
        'dis_scaler': dis_scaler,
        'od_scaler': od_scaler,
    }


def evaluate(trained, test_areas, data_path, device=None):
    """Evaluate NetGAN on test areas."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = trained['generator']
    nfeat_scaler = trained['nfeat_scaler']
    dis_scaler = trained['dis_scaler']
    od_scaler = trained['od_scaler']

    nf_te, adj_te, dis_te, od_te = load_graph_data(test_areas, data_path)

    generator.eval()
    metrics_all = []
    for nf, adj, dis, od in zip(nf_te, adj_te, dis_te, od_te):
        nf_s = nfeat_scaler.transform(nf)
        dis_s = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
        nf_t = torch.FloatTensor(nf_s).to(device)
        dis_t = torch.FloatTensor(dis_s).to(device)
        g = build_dgl_graph(adj, device)
        with torch.no_grad():
            od_gen, _, _ = generator.generate_OD_net(g, nf_t, dis_t)
        od_hat = od_scaler.inverse_transform(od_gen.cpu().numpy().reshape(-1, 1)).reshape(od.shape)
        od_hat[od_hat < 0] = 0
        metrics_all.append(canonical_od_metrics(od_hat, od))

    return metrics_all
