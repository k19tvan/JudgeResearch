"""Problem 08: Transformer Encoder Layer"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer: Self-Attention + Feed-Forward Network."""
    
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, src, pos, src_key_padding_mask=None):
        """Forward pass with pre-norm architecture."""
        # Pre-LN self-attention with positional injection into q and k
        norm = self.norm1(src)
        q = k = norm + pos
        attn_out, _ = self.attn(q, k, norm, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_out)

        # Pre-LN FFN
        norm2 = self.norm2(src)
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(norm2))))
        return src + self.dropout(ffn_out)
