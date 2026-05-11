"""
TRF block for the model
"""

import torch.nn as nn
from src.components.multihead_attention import MultiHeadAttention
from src.components.layer_norm import LayerNorm
from src.components.ffn import FeedForward

class TransformerBlock(nn.Module):
    """
    TRF block class
    """
    def __init__(self, config):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_len"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"]
        )

        self.ffn = FeedForward(config=config)

        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])

        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)

        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)

        x = shortcut + x

        return x