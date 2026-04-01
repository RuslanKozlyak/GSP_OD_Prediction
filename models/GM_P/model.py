import torch
import torch.nn as nn

class GRAVITY(nn.Module):
    def __init__(self):
        super(GRAVITY, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(1.0))  # was 0.5 → caused exp(-69300)≈0 with raw distances
        self.G     = nn.Parameter(torch.tensor(1.0))  # was randn → could be negative/large

    def forward(self, x):
        x = x + 1e-10
        logy = (self.alpha * torch.log(x[:, 0]) +
                self.beta  * torch.log(x[:, 1]) +
                x[:, 2]    * torch.log(self.gamma ** 2 + 1e-10))
        return self.G.abs() * torch.exp(logy.clamp(-50, 50))