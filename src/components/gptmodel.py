"""
Class that contains all the logics of the model
"""
import torch
import torch.nn as nn

from src.components.transformer_block import TransformerBlock
from src.components.layer_norm import LayerNorm

class GPTModel(nn.Module):
    """
    Class that encapsulates all the components of the model
    """
    def __init__(self, config):
        super().__init__()

        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_len"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])

        self.norm = LayerNorm(config["emb_dim"])
        self.output_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape

        tok_embs = self.tok_emb(in_idx)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embs + pos_embs
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.norm(x)

        logits = self.output_head(x)

        return logits