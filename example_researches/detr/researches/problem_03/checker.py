# learning_path/problem_03/checker.py
import torch
from starter import PositionEmbeddingSine

def check_pe():
    B, H, W = 2, 5, 5
    num_pos_feats = 128
    mask = torch.zeros((B, H, W), dtype=torch.bool)
    
    module = PositionEmbeddingSine(num_pos_feats=num_pos_feats, normalize=True)
    out = module(mask)
    C = num_pos_feats * 2
    
    assert out.shape == (B, C, H, W), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN found in output"
    assert out.dtype == torch.float32, "Must be float32"
    
    print("All Problem 03 checks passed")

if __name__ == "__main__":
    check_pe()
