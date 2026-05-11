"""
Activation function for the ffn inside of the model
"""

import torch
import torch.nn as nn

class GELU(nn.Module):
    """
    GELU activation function
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        gelu = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        return gelu