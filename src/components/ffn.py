"""
FFN for the model
"""

import torch.nn as nn
from src.components.gelu import GELU


class FeedForward(nn.Module):
    """
    FFN with GELU activation function
    """

    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
