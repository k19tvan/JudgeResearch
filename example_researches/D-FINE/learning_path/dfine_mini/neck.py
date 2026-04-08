"""Multi-Level Feature Pyramid Neck for D-FINE.

Transforms a single-level backbone feature into multi-level features
for processing objects at different scales. Similar to FPN (Feature Pyramid Network).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNeck(nn.Module):
    """Simple FPN-style feature pyramid neck.
    
    Input: (B, C_in, H, W) single-level feature
    Output: List of (B, C_out, H_i, W_i) multi-level features
    
    Creates 3 levels via:
    - Level 1: Input feature (1x scale)
    - Level 2: Downsampled 2x (1/2 scale)
    - Level 3: Downsampled 4x (1/4 scale)
    """
    
    def __init__(self, c_in=96, c_out=256, num_levels=3, use_relu=True):
        """
        Args:
            c_in: Input channel dimension (from backbone)
            c_out: Output channel dimension for all levels
            num_levels: Number of pyramid levels to create
            use_relu: Whether to use ReLU activation
        """
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_levels = num_levels
        
        # Channel projection from backbone to uniform size
        self.proj = nn.Conv2d(c_in, c_out, 1)
        
        # Downsampling layers for creating pyramid levels
        # Level i: stride = 2^(i-1)
        self.downsample_layers = nn.ModuleList()
        for i in range(1, num_levels):
            # Simple stride-2 conv downsampler
            down = nn.Sequential(
                nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=use_relu) if use_relu else nn.Identity()
            )
            self.downsample_layers.append(down)
        
        self.act = nn.ReLU(inplace=use_relu) if use_relu else nn.Identity()
    
    def forward(self, backbone_feat):
        """
        Args:
            backbone_feat: (B, C_in, H, W)
        
        Returns:
            features: List[(B, C_out, H_i, W_i)] for i in 0..num_levels-1
        """
        # Project to uniform channel dimension
        feat = self.proj(backbone_feat)  # (B, C_out, H, W)
        feat = self.act(feat)
        
        # Collect features at each pyramid level
        features = [feat]  # Level 0: full resolution
        
        # Create downsampled levels
        current_feat = feat
        for down_layer in self.downsample_layers:
            current_feat = down_layer(current_feat)
            features.append(current_feat)
        
        return features


class FPNNeck(nn.Module):
    """Improved FPN-style neck with lateral connections.
    
    Creates multi-level features with top-down refinement:
    - Extract features at multiple scales from backbone
    - Apply lateral projections to align channels
    - Top-down refinement: upsample high-level features and merge with lower levels
    
    More sophisticated than SimpleNeck but still educational.
    """
    
    def __init__(self, c_in=96, c_out=256, num_levels=3, use_relu=True):
        """
        Args:
            c_in: Input channel dimension
            c_out: Output channel dimension
            num_levels: Number of pyramid levels
            use_relu: Use ReLU activation
        """
        super().__init__()
        self.num_levels = num_levels
        self.c_out = c_out
        
        # Bottom-up pathway: produce P3, P4, P5 from backbone
        # We'll create them via max pooling downsampling
        self.proj = nn.Conv2d(c_in, c_out, 1)
        
        # Downsampling path (bottom-up)
        self.downsample_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(num_levels - 1)
        ])
        
        # Lateral layers (1x1 convs) to project all features to C_out channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c_out, c_out, 1) for _ in range(num_levels)
        ])
        
        # Smooth layers (3x3 convs) after top-down merging
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(c_out, c_out, 3, padding=1) for _ in range(num_levels)
        ])
        
        self.act = nn.ReLU(inplace=use_relu) if use_relu else nn.Identity()
    
    def forward(self, backbone_feat):
        """
        Args:
            backbone_feat: (B, C_in, H, W)
        
        Returns:
            features: List[(B, C_out, H_i, W_i)] for i in 0..num_levels-1
        """
        # Project input to uniform channels
        feat_c3 = self.proj(backbone_feat)  # (B, C_out, H, W)
        feat_c3 = self.act(feat_c3)
        
        # Bottom-up: create coarser levels by downsampling
        features_bu = [feat_c3]
        current_feat = feat_c3
        for i, down in enumerate(self.downsample_layers):
            current_feat = down(current_feat)  # Max pool 2x2
            features_bu.append(current_feat)
        
        # Top-down merging with lateral connections
        # Start from coarsest level and merge with finer levels
        features_td = [features_bu[-1]]  # Start with coarsest
        
        for i in range(len(features_bu) - 2, -1, -1):
            # Upsample coarser feature and merge with finer feature
            upsampled = F.interpolate(
                features_td[-1], 
                size=features_bu[i].shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            # Lateral connection: merge upsampled with skip connection
            merged = upsampled + self.lateral_convs[i](features_bu[i])
            merged = self.act(merged)
            # Smooth
            refined = self.fpn_convs[i](merged)
            refined = self.act(refined)
            features_td.append(refined)
        
        # Reverse to match pyramid order (coarse to fine)
        features_td.reverse()
        
        return features_td
