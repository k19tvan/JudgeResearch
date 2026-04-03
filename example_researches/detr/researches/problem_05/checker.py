# learning_path/problem_05/checker.py
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starter import TransformerEncoderLayer

# Mocking the attention to avoid missing dependencies from prob04
class MockAttn(nn.Module):
    def forward(self, q, k, v, pos_q, pos_k):
        return v * 0.1 # Mock

def check_encoder_layer():
    module = TransformerEncoderLayer()
    module.self_attn = MockAttn() # Inject Mock
    
    N, B, C = 10, 2, 256
    src = torch.rand(N, B, C)
    pos = torch.rand(N, B, C)
    
    try:
        out = module(src, pos)
        assert out.shape == (N, B, C), f"Output shape mismatch: {out.shape}"
        print("All Problem 05 checks passed")
    except NotImplementedError:
        print("Implement the Encoder Layer to pass.")
        raise
    
if __name__ == "__main__":
    check_encoder_layer()
