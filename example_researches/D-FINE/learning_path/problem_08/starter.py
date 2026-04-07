import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention helper module.
    """
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        raise NotImplementedError


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer: Self-Attention + Feed-Forward Network
    Uses pre-norm architecture: LayerNorm → Attention → Residual
    Positional encodings are added to queries and keys.
    
    Args:
        d_model: Embedding dimension
        num_heads: Number of attention heads
        d_ffn: Dimension of feed-forward network hidden layer
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, src, pos, src_key_padding_mask=None):
        """
        Args:
            src: Source token sequence of shape (B, T, d_model)
            pos: Positional encodings of shape (B, T, d_model) to add to q and k
            src_key_padding_mask: Optional mask of shape (B, T), bool
        
        Returns:
            Output tensor of shape (B, T, d_model)
        """
        raise NotImplementedError
