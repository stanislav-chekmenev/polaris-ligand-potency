import torch

import config as cfg

from models.attention.attention import SelfAttention


class TransformerBlock(torch.nn.Module):
    """
    TransformerBlock class.
    This class is used to implement a transformer block with self-attention and layer normalization.
    The input features are the atom features, the molecular features, 3d mace features and the barycenters.
    """

    def __init__(self):
        """
        Initialize the TransformerBlock class.
        """
        super().__init__()
        self.attention = SelfAttention()
        self.layernorm = torch.nn.LayerNorm(cfg.IN_ATTENTION_DIM)
        self.act = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.attention(x)
        x = self.layernorm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x + x_skip
        return x
