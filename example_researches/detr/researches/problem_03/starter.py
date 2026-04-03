# learning_path/problem_03/starter.py
import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: (B, H, W) boolean mask tensor 
        Returns:
            pos_encoding: (B, C, H, W) where C = 2 * num_pos_feats
        """
        raise NotImplementedError("Implement 2D Sine sequence positional encoding")
