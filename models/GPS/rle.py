import numpy as np
import torch
import torch.nn as nn


class RelativeLocationEncoder(nn.Module):
    """
    RLE: кодирует относительное положение (origin - destination) в векторное пространство.
    Использует многомасштабные sinusoidal функции — Space2Vec из статьи.
    Два набора базисных векторов (RLE вариант из статьи) для устранения гексагонального артефакта.
    """
    def __init__(self, freq=16, lambda_min=1.0, lambda_max=20000.0, out_dim=64):
        super().__init__()
        self.freq = freq
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.out_dim = out_dim

        # Базисные векторы: 3 вектора под углом 2pi/3 — первый набор
        angles1 = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        self.register_buffer('basis1', torch.FloatTensor(
            [[np.cos(a), np.sin(a)] for a in angles1]
        ))  # (3, 2)

        # Второй набор — повёрнут на pi/6 для устранения гексагонального артефакта
        angles2 = [a + np.pi / 6 for a in angles1]
        self.register_buffer('basis2', torch.FloatTensor(
            [[np.cos(a), np.sin(a)] for a in angles2]
        ))  # (3, 2)

        # Логарифмически равномерно распределённые масштабы
        g = lambda_max / lambda_min
        self.register_buffer('scales', torch.FloatTensor(
            [lambda_min * (g ** (s / max(freq - 1, 1))) for s in range(freq)]
        ))  # (freq,)

        # Размерность входа в FF: 2 набора x 3 вектора x freq x 2 (sin/cos) = 12*freq
        pe_dim = 2 * 3 * freq * 2
        self.ff = nn.Sequential(
            nn.Linear(pe_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim)
        )

    def _encode_basis(self, rel_loc, basis):
        """Кодирует rel_loc через набор базисных векторов."""
        # rel_loc: (B, 2), basis: (3, 2), scales: (freq,)
        # Проекции: (B, 3)
        proj = rel_loc @ basis.T  # (B, 3)

        # Многомасштабное кодирование: (B, 3, freq)
        proj_scaled = proj.unsqueeze(2) / self.scales.view(1, 1, -1)  # (B, 3, freq)

        # sin и cos: (B, 3, freq, 2)
        pe = torch.stack([torch.cos(proj_scaled), torch.sin(proj_scaled)], dim=-1)
        return pe.reshape(rel_loc.shape[0], -1)  # (B, 3*freq*2)

    def forward(self, rel_loc):
        """rel_loc: (B, 2) — вектор (coord_origin - coord_destination)"""
        pe1 = self._encode_basis(rel_loc, self.basis1)  # (B, 3*freq*2)
        pe2 = self._encode_basis(rel_loc, self.basis2)  # (B, 3*freq*2)
        pe = torch.cat([pe1, pe2], dim=-1)               # (B, 6*freq*2)
        return self.ff(pe)                                # (B, out_dim)
