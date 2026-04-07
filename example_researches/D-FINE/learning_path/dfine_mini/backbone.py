"""Problem 10: HGNetV2 Backbone Stem"""
import torch
import torch.nn as nn


def cbr(in_c, out_c, k=3, s=1, p=1, g=1):
    """Conv-BatchNorm-ReLU building block."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class HGNetV2Stem(nn.Module):
    """HGNetV2 Stem: Initial downsampling stage with 2 strided convolutions + LCS block."""
    
    def __init__(self):
        super().__init__()
        # Stem: 2x stride-2 convolutions
        self.stem = nn.Sequential(
            cbr(3, 24, k=3, s=2, p=1),
            cbr(24, 48, k=3, s=2, p=1),
        )
        # LCS Block: PW → DW(stride=2) → PW
        self.lcs = nn.Sequential(
            cbr(48, 96, k=1, s=1, p=0),            # pointwise expand
            cbr(96, 96, k=3, s=2, p=1, g=96),      # depthwise stride-2
            cbr(96, 96, k=1, s=1, p=0),             # pointwise refine
        )

    def forward(self, x):
        """Forward pass through stem and LCS block."""
        stem_out = self.stem(x)   # (B, 48, H/4, W/4)
        lcs_out  = self.lcs(stem_out)   # (B, 96, H/8, W/8)
        return stem_out, lcs_out
