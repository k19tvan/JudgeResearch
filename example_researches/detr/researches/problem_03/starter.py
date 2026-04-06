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
        
        B, H, W = mask.shape
        C = 2 * self.num_pos_feats
        
        mask = ~mask # B, H, W
        y_embed = mask.cumsum(dim=1, dtype=torch.float32) # B, H, W
        x_embed = mask.cumsum(dim=2, dtype=torch.float32) # B, H, W
        
        y_embed = y_embed / y_embed[:, -1:, :] # B, H, W
        x_embed = x_embed / x_embed[:, :, -1:] # B, H, W
        
        if self.normalize:
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale
            
        dim_t = torch.arange(C // 4, dtype=torch.float32) # (C / 4, )
        dim_t = self.temperature ** (2 * dim_t / (C / 2))
        
        p_y = y_embed[..., None] / dim_t # (B, H, W, C / 4)
        p_x = x_embed[..., None] / dim_t # (B, H, W, C / 4)
        
        pe_y = torch.stack([p_y.sin(), p_y.cos()], dim=-1) # (B, H, W, C / 4, 2)
        pe_y = pe_y.flatten(-2) # (B, H, W, C / 2)
        
        pe_x = torch.stack([p_x.sin(), p_x.cos()], dim=-1) # (B, H, W, C / 4, 2)
        pe_x = pe_x.flatten(-2) # (B, H, W, C / 2)

        pe = torch.cat([pe_x, pe_y], dim=-1) # B, H, W, C
        pe = pe.permute(0, 3, 1, 2) # B, C, H, W
        
        return pe
        
        raise NotImplementedError("Implement 2D Sine sequence positional encoding")
