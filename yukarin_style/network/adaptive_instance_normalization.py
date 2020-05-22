from torch import Tensor, nn
import torch


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, feature_size: int, style_size: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(feature_size, affine=False)
        self.linear = nn.Linear(style_size, feature_size * 2)

    def forward(self, x: Tensor, s: Tensor):
        h = self.linear(s)
        h = h.view(h.shape[0], h.shape[1], 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
