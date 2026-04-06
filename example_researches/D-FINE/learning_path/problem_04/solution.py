import torch
import torch.nn as nn
from typing import List

class HybridEncoderProjectors(nn.Module):
    """
    Problem 04: Maps multi-scale CNN backbones to a uniform hidden dimension.
    """
    def __init__(self, in_channels: List[int] = [512, 1024, 2048], hidden_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.ModuleList()
        for in_c in in_channels:
            proj = nn.Sequential(
                nn.Conv2d(in_c, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim)
            )
            self.input_proj.append(proj)
            
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of tensors from backbone. E.g. [ (B, 512, H1, W1), (B, 1024, H2, W2), (B, 2048, H3, W3) ]
        Returns:
            List of projected tensors, all with `hidden_dim` channels.
        """
        assert len(features) == len(self.input_proj), "Mismatched number of features and projectors"
        
        out_features = []
        for i, proj in enumerate(self.input_proj):
            out_features.append(proj(features[i]))
            
        return out_features
