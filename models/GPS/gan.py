import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from .model import restore_force_noise, set_force_noise


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class ODSequenceDiscriminator(nn.Module):
    """TCN discriminator over OD random-walk sequences.

    Each step is the destination node feature vector concatenated with the
    walked edge flow, matching the sequence signal used by ODGN-style GANs.
    """

    def __init__(self, input_dim, hidden_dim=64, n_layers=4, kernel_size=5, dropout=0.05):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tcn = nn.Sequential(*[
            TemporalBlock(
                hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(n_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        x = self.input_proj(seq).transpose(1, 2)
        x = self.tcn(x)
        return self.head(x[:, :, -1]).squeeze(-1)


def generated_od_matrix(model, cd, config):
    """Differentiably decode a full train-scale OD matrix."""
    ne = model.encode(cd['graph_data'])
    n_nodes = cd['num_nodes']
    dest_idx = torch.arange(n_nodes, device=ne.device, dtype=torch.long)
    outflow = torch.as_tensor(cd['outflow_train'], dtype=torch.float32, device=ne.device)
    rows = []
    for oi in range(n_nodes):
        scores = model.decode_row(
            ne,
            oi,
            dest_idx,
            cd['distance_matrix'],
            coords=cd.get('coords_tensor'),
        )
        rows.append(_scores_to_flow(scores, outflow[oi], config))
    return torch.stack(rows, dim=0)


def sample_walk_sequences(
    flow_matrix,
    node_features,
    walk_len,
    batch_size,
    flow_scale=None,
    tau=1.0,
    hard=True,
    eps=1e-8,
):
    """Sample differentiable random-walk sequences from a weighted OD matrix."""
    flow_matrix = torch.nan_to_num(flow_matrix.float(), nan=0.0, posinf=1e6, neginf=0.0)
    flow_matrix = flow_matrix.clamp_min(0.0)
    node_features = node_features.to(flow_matrix.device).float()
    n_nodes = flow_matrix.size(0)
    if n_nodes < 1:
        raise ValueError("Cannot sample OD walks from an empty matrix")

    if flow_scale is None:
        flow_scale = torch.log1p(flow_matrix.detach().max()).clamp_min(1.0)
    elif not torch.is_tensor(flow_scale):
        flow_scale = torch.tensor(float(flow_scale), device=flow_matrix.device)
    flow_scale = flow_scale.to(flow_matrix.device).clamp_min(1.0)

    current_idx = torch.randint(n_nodes, (batch_size,), device=flow_matrix.device)
    current = F.one_hot(current_idx, num_classes=n_nodes).float()
    uniform_logits = torch.zeros(batch_size, n_nodes, device=flow_matrix.device)
    steps = []

    for _ in range(walk_len):
        row_flow = current @ flow_matrix
        row_sum = row_flow.sum(dim=-1, keepdim=True)
        logits = torch.where(row_sum > eps, torch.log(row_flow + eps), uniform_logits)
        next_node = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        edge_flow = (row_flow * next_node).sum(dim=-1, keepdim=True)
        dest_features = next_node @ node_features
        flow_feature = torch.log1p(edge_flow.clamp_min(0.0)) / flow_scale
        steps.append(torch.cat([dest_features, flow_feature], dim=-1))
        current = next_node

    return torch.stack(steps, dim=1)


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1.0 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def gan_step_for_city(model, discriminator, cd, config, generator_optimizer, discriminator_optimizer, epoch=None):
    """Run one ODGN-style WGAN update for a city."""
    node_features = cd['graph_data'].x.detach()
    real_od = torch.as_tensor(cd['od_matrix_train'], dtype=torch.float32, device=node_features.device)
    if real_od.sum().item() <= 0:
        return {'gan_d_loss': float('nan'), 'gan_g_loss': float('nan'), 'gan_gp': float('nan')}

    flow_scale = _real_flow_scale(cd, node_features.device)
    d_losses = []
    gp_losses = []

    discriminator.train()
    _set_requires_grad(discriminator, True)
    n_critic = _effective_n_critic(config, epoch)
    # Fresh fake OD samples are generated inside the loop below.
    # The legacy cached-matrix comment below is no longer accurate.
    # Decode the generated OD matrix once for all critic steps — the generator
    # weights do not change during discriminator updates, so reusing the same
    # matrix is equivalent to sampling a fresh one each step. Walk sequences
    # are still resampled independently every critic iteration.
    for _ in range(n_critic):
        discriminator_optimizer.zero_grad()
        fake_od = _detached_generated_od_matrix(model, cd, config)
        real_seq = sample_walk_sequences(
            real_od,
            node_features,
            config.gan_walk_len,
            config.gan_walk_batch_size,
            flow_scale=flow_scale,
            tau=config.gan_tau,
            hard=True,
        )
        fake_seq = sample_walk_sequences(
            fake_od,
            node_features,
            config.gan_walk_len,
            config.gan_walk_batch_size,
            flow_scale=flow_scale,
            tau=config.gan_tau,
            hard=True,
        )
        real_score = discriminator(real_seq)
        fake_score = discriminator(fake_seq)
        if config.gan_regularizer == 'gp':
            gp = compute_gradient_penalty(discriminator, real_seq, fake_seq)
            loss_d = fake_score.mean() - real_score.mean() + config.gan_gp_lambda * gp
        else:
            gp = torch.zeros((), device=real_seq.device)
            loss_d = fake_score.mean() - real_score.mean()
        if torch.isnan(loss_d) or torch.isinf(loss_d):
            continue
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        discriminator_optimizer.step()
        if config.gan_regularizer == 'clip':
            _clip_weights(discriminator, config.gan_clip_value)
        d_losses.append(float(loss_d.detach().cpu()))
        gp_losses.append(float(gp.detach().cpu()))

    _set_requires_grad(discriminator, False)
    generator_optimizer.zero_grad()
    model.train()
    fake_od = generated_od_matrix(model, cd, config)
    fake_seq = sample_walk_sequences(
        fake_od,
        node_features,
        config.gan_walk_len,
        config.gan_walk_batch_size,
        flow_scale=flow_scale,
        tau=config.gan_tau,
        hard=True,
    )
    loss_g_raw = -discriminator(fake_seq).mean()
    loss_g = config.adv_weight * loss_g_raw
    if not (torch.isnan(loss_g) or torch.isinf(loss_g)):
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        generator_optimizer.step()
        g_loss_value = float(loss_g_raw.detach().cpu())
    else:
        g_loss_value = float('nan')
    _set_requires_grad(discriminator, True)

    return {
        'gan_d_loss': float(np.mean(d_losses)) if d_losses else float('nan'),
        'gan_g_loss': g_loss_value,
        'gan_gp': float(np.mean(gp_losses)) if gp_losses else float('nan'),
    }


def _scores_to_flow(scores, outflow, config):
    pm = config.prediction_mode
    use_log_flow = config.use_log_transform and config.loss_type in ('huber', 'multitask', 'mae')
    use_log_norm = use_log_flow and pm == 'normalized'
    is_gravity = getattr(config, 'decoder_type', None) == 'gravity_guided'

    if config.loss_type == 'zinb':
        flow = F.softplus(scores)
    elif pm == 'normalized':
        if use_log_norm:
            prob = torch.sigmoid(scores)
            flow = torch.expm1(prob * torch.log1p(outflow.clamp_min(0.0)))
        else:
            flow = F.softmax(scores, dim=0) * outflow.clamp_min(0.0)
    else:
        if is_gravity:
            flow = F.softplus(scores) if use_log_flow else torch.exp(scores.clamp(max=20.0))
        else:
            flow = F.relu(scores)
        if use_log_flow:
            flow = torch.expm1(flow.clamp_min(0.0))

    return torch.nan_to_num(flow, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)


def _detached_generated_od_matrix(model, cd, config):
    was_training = model.training
    model.eval()
    noise_states = set_force_noise(model, True)
    try:
        with torch.no_grad():
            fake_od = generated_od_matrix(model, cd, config).detach()
    finally:
        restore_force_noise(noise_states)
    if was_training:
        model.train()
    return fake_od


def _real_flow_scale(cd, target_device):
    od = cd.get('od_matrix_train')
    max_flow = float(np.nanmax(od)) if od is not None and np.size(od) else 1.0
    if not np.isfinite(max_flow) or max_flow < 0:
        max_flow = 1.0
    return torch.tensor(max(np.log1p(max_flow), 1.0), dtype=torch.float32, device=target_device)


def _set_requires_grad(module, enabled):
    for param in module.parameters():
        param.requires_grad_(enabled)


def _effective_n_critic(config, epoch):
    switch_epoch = getattr(config, 'gan_n_critic_after_epoch', 0)
    if switch_epoch and epoch is not None and epoch > switch_epoch:
        return config.gan_n_critic_after
    return config.gan_n_critic


def _clip_weights(module, clip_value):
    with torch.no_grad():
        for param in module.parameters():
            param.clamp_(-clip_value, clip_value)
