"""
Layer normalization class for the model
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Normalization class
    """

    def __init__(self, emb_dim):
        super().__init__()

        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized_x = (x - mean) / torch.sqrt(var + self.epsilon)

        return self.scale * normalized_x + self.shift
