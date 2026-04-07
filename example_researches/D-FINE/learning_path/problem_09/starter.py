import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MHA(nn.Module):
    """Multi-Head Attention helper."""
    def __init__(self, d, h, dr=0.0):
        super().__init__()
        self.h = h
        self.dk = d // h
        self.sc = math.sqrt(self.dk)
        self.d = d
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.drop = nn.Dropout(dr)

    def forward(self, q, k, v, mask=None):
        raise NotImplementedError


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with D-FINE Target Gating.
    Components: Self-Attention → Cross-Attention with Target Gating → FFN
    Target Gating replaces the standard residual connection after cross-attention
    with learned sigmoid gates for dynamic context blending.
    
    Args:
        d_model: Embedding dimension
        num_heads: Number of attention heads
        d_ffn: Dimension of feed-forward network hidden layer
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.sa = MHA(d_model, num_heads, dropout)
        self.ca = MHA(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Target Gating Layer parameters
        self.gate1 = nn.Linear(2*d_model, d_model)
        self.gate2 = nn.Linear(2*d_model, d_model)
        # FFN
        self.ff1 = nn.Linear(d_model, d_ffn)
        self.ff2 = nn.Linear(d_ffn, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_query_pos, memory_pos, memory_key_padding_mask=None):
        """
        Args:
            tgt: Target (object queries) of shape (B, N, d_model)
            memory: Encoder output of shape (B, T, d_model)
            tgt_query_pos: Query positional embeddings of shape (B, N, d_model)
            memory_pos: Memory positional encodings of shape (B, T, d_model)
            memory_key_padding_mask: Optional mask of shape (B, T), bool
        
        Returns:
            Output tensor of shape (B, N, d_model)
        """
        raise NotImplementedError
