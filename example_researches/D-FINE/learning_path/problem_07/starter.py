import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: allows attending to information from different representation subspaces.
    Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        d_model: Embedding dimension
        num_heads: Number of attention heads (must divide d_model evenly)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None):
        """
        Args:
            q: Query tensor of shape (B, T_q, d_model)
            k: Key tensor of shape (B, T_k, d_model)
            v: Value tensor of shape (B, T_k, d_model)
            key_padding_mask: Optional mask of shape (B, T_k), bool, True for positions to ignore
        
        Returns:
            out: Attention output of shape (B, T_q, d_model)
            attn_weights: Attention weights of shape (B, num_heads, T_q, T_k)
        """
        raise NotImplementedError
