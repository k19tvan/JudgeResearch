# learning_path/problem_04/checker.py
import torch
from starter import MultiHeadAttentionWithPos

def check_attention():
    N, B, C = 20, 2, 64
    heads = 4
    module = MultiHeadAttentionWithPos(embed_dim=C, num_heads=heads)
    
    q = torch.rand(N, B, C)
    k = torch.rand(N, B, C)
    v = torch.rand(N, B, C)
    pos_q = torch.rand(N, B, C)
    pos_k = torch.rand(N, B, C)
    
    try:
        out = module(q, k, v, pos_q, pos_k)
        assert out.shape == (N, B, C), f"Output shape mismatch: {out.shape}"
        print("All Problem 04 checks passed")
    except NotImplementedError:
        print("Implement the Attention mechanism to pass.")
        raise

if __name__ == "__main__":
    check_attention()
