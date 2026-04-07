import torch
import torch.nn as nn
import math


class PositionEmbeddingSine2D(nn.Module):
    """
    2D sinusoidal positional encodings for transformer models.
    Encodes spatial position (H, W) using sin/cos functions at different frequencies.
    
    Args:
        d_model: Dimension of the embedding (must be even)
        temperature: Base temperature for frequency calculation (default: 10000.0)
        normalize: Whether to scale positions to [0, 2π] (default: True)
    """
    
    def __init__(self, d_model=256, temperature=10000.0, normalize=True):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map of shape (B, C, H, W)
        
        Returns:
            Positional encodings of shape (B, H*W, d_model)
        """
        raise NotImplementedError
