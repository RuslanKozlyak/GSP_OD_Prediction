import time
import os
import sys

import numpy as np
import torch

from models.shared.metrics import cal_od_metrics, average_listed_metrics
from models.shared.data_load import load_graph_data, get_scalers, build_dgl_graph


def train(train_areas, valid_areas, data_path,
          device=None, nfeat_scaler=None, dis_scaler=None, od_scaler=None,
          n_epochs=2):
    """Train NetGAN (GAT + Generator + Discriminator).

    Args:
        train_areas: list of area IDs for training
        valid_areas: list of area IDs for validation
        data_path: path to data root
        device: torch device
        nfeat_scaler, dis_scaler, od_scaler: pre-fitted scalers
        n_epochs: number of GAN training epochs

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
    if nfeat_scaler is None or dis_scaler is None or od_scaler is None:
        nf_v, _, dis_v, od_v = load_graph_data(train_areas, data_path)
        nfeat_scaler, dis_scaler, od_scaler = get_scalers(nf_v, dis_v, od_v)

    nf_tr, adj_tr, dis_tr, od_tr = load_graph_data(train_areas, data_path)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=3e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=3e-4)

    print(f'  NetGAN: training for {n_epochs} epochs...')
    t0 = time.time()
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        for nf, adj, dis, od in zip(nf_tr, adj_tr, dis_tr, od_tr):
            nf_s = nfeat_scaler.transform(nf)
            dis_s = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)

            nf_t = torch.FloatTensor(nf_s).to(device)
            g = build_dgl_graph(adj, device)
            dis_t = torch.FloatTensor(dis_s).to(device)

            opt_g.zero_grad()
            fake_batch = generator.sample_generated_batch(g, nf_t, dis_t, 128).to(device)
            loss_g = -torch.mean(discriminator(fake_batch))
            loss_g.backward()
            opt_g.step()

            if epoch % 5 == 0:
                opt_d.zero_grad()
                with torch.no_grad():
                    _, adjacency, logp = generator.generate_OD_net(g, nf_t, dis_t)
                    batch = [sample_one_random_walk(adjacency, logp) for _ in range(128)]
                    fake_batch = torch.stack(batch).to(device)

                od_s = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
                real_batch = torch.FloatTensor(sample_batch_real(od_s)).to(device)
                loss_d = (torch.mean(discriminator(fake_batch))
                          - torch.mean(discriminator(real_batch))
                          + 10 * compute_gradient_penalty(discriminator, real_batch, fake_batch))
                loss_d.backward()
                opt_d.step()

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
        od_hat = od_scaler.inverse_transform(od_gen.cpu().numpy())
        od_hat[od_hat < 0] = 0
        metrics_all.append(cal_od_metrics(od_hat, od))

    return metrics_all


if __name__ == '__main__':
    from pprint import pprint
    from models.shared.data_load import SINGLE_CITY_ID, DEFAULT_DATA_PATH

    data_path = str(DEFAULT_DATA_PATH)
    areas = [SINGLE_CITY_ID]

    trained = train(areas, areas, data_path)
    metrics_all = evaluate(trained, areas, data_path)
    pprint(average_listed_metrics(metrics_all))
