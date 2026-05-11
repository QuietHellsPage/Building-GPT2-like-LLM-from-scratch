"""
Multihead attention class with optimized operations
"""

import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    """
    Main class to encapsulate MultiHeadAttention
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (
            d_out % num_heads == 0
        ), "Output dimension has to be divisible by number of heads"

        self.d_out = d_out
        self.num_heads = num_heads

        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.output_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)

        bool_mask = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(bool_mask, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)

        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)

        context_vector = self.output_proj(context_vector)

        return context_vector
