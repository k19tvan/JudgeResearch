import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SimplifiedBiFPNLayer(nn.Module):
    """
    Problem 05: Simplified Top-Down and Bottom-Up cross-scale interaction.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # Top-Down FPN Convs
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        ])
        
        self.fpn_blocks = nn.ModuleList([
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        ])
        
        # Bottom-Up PAN Convs
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        ])
        
        self.pan_blocks = nn.ModuleList([
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        features: [P3, P4, P5] (low to high level / high to low res)
        all with 'hidden_dim' channels.
        """
        [p3, p4, p5] = features
        
        # --- Top-Down Pathway ---
        # fpn_p5 = p5 
        # fpn_p4 = p4 + upsample(p5)
        # fpn_p3 = p3 + upsample(fpn_p4)
        
        # P5 -> P4
        fpn_p4 = p4 + self.lateral_convs[0](p4)
        up_p5 = F.interpolate(p5, size=fpn_p4.shape[-2:], mode="nearest")
        fpn_p4 = self.fpn_blocks[0](torch.cat([fpn_p4, up_p5], dim=1))
        
        # P4 -> P3
        fpn_p3 = p3 + self.lateral_convs[1](p3)
        up_p4 = F.interpolate(fpn_p4, size=fpn_p3.shape[-2:], mode="nearest")
        fpn_p3 = self.fpn_blocks[1](torch.cat([fpn_p3, up_p4], dim=1))
        
        # --- Bottom-Up Pathway ---
        # pan_p3 = fpn_p3
        # pan_p4 = fpn_p4 + downsample(pan_p3)
        # pan_p5 = p5 + downsample(pan_p4)
        
        pan_p3 = fpn_p3
        
        # P3 -> P4
        down_p3 = self.downsample_convs[0](pan_p3)
        pan_p4 = self.pan_blocks[0](torch.cat([fpn_p4, down_p3], dim=1))
        
        # P4 -> P5
        down_p4 = self.downsample_convs[1](pan_p4)
        pan_p5 = self.pan_blocks[1](torch.cat([p5, down_p4], dim=1))
        
        return [pan_p3, pan_p4, pan_p5]
