import torch
import torch.nn as nn
import math


class PositionEmbeddingSine2D(nn.Module):
    def __init__(self, d_model=256, temperature=10000.0, normalize=True):
        super().__init__()
        self.d_model     = d_model
        self.temperature = temperature
        self.normalize   = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device      = x.device
        dtype       = x.dtype

        # Normalized position grids in [0, 1]
        y_grid = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W) / H  # (H, W)
        x_grid = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W) / W  # (H, W)

        # Frequency indices: 0,1,...,d_model//4 - 1  (each produces sin+cos → d_model//2 per axis)
        half = self.d_model // 2
        quarter = half // 2
        dim_t = torch.arange(quarter, device=device, dtype=dtype)
        omega = self.temperature ** (2 * dim_t / self.d_model)  # (quarter,)

        # Scale: multiply grids by 2*pi if normalize
        scale = 2 * math.pi if self.normalize else 1.0
        arg_x = scale * x_grid.unsqueeze(-1) / omega   # (H, W, quarter)
        arg_y = scale * y_grid.unsqueeze(-1) / omega   # (H, W, quarter)

        # Interleave sin, cos → (H, W, half)
        pos_x = torch.stack([arg_x.sin(), arg_x.cos()], dim=-1).flatten(-2)  # (H, W, half)
        pos_y = torch.stack([arg_y.sin(), arg_y.cos()], dim=-1).flatten(-2)  # (H, W, half)

        # Concat and expand batch
        pos = torch.cat([pos_x, pos_y], dim=-1)         # (H, W, d_model)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)    # (B, H, W, d_model)
        return pos.flatten(1, 2)                         # (B, H*W, d_model)
