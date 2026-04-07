"""Problem 07: Multi-Head Self-Attention"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention: allows attending to information from different representation subspaces."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.scale     = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, H, T, d_k)"""
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, key_padding_mask=None):
        """Compute multi-head attention."""
        B, T_q, _ = q.shape
        # Project
        Q = self._split_heads(self.W_q(q))   # (B, H, T_q, d_k)
        K = self._split_heads(self.W_k(k))   # (B, H, T_k, d_k)
        V = self._split_heads(self.W_v(v))   # (B, H, T_k, d_k)

        # Scaled dot-product
        scores = Q @ K.transpose(-2, -1) / self.scale   # (B, H, T_q, T_k)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = scores.softmax(dim=-1)
        attn_drop    = self.dropout(attn_weights)

        # Context
        ctx = attn_drop @ V                                              # (B, H, T_q, d_k)
        ctx = ctx.transpose(1, 2).contiguous().reshape(B, T_q, self.d_model)
        return self.W_o(ctx), attn_weights
