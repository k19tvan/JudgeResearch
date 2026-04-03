# learning_path/problem_04/starter.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionWithPos(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # In actual PyTorch, nn.MultiheadAttention does this internally,
        # but you should wrap it here to correctly add positional encodings.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                query_pos: torch.Tensor = None, key_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q, k, v: (N, B, C)
            query_pos: (N, B, C) Optional positional encoding for Q
            key_pos: (N, B, C) Optional positional encoding for K
        Returns:
            output: (N, B, C)
        """
        raise NotImplementedError("Implement Attention with Positional injection")
