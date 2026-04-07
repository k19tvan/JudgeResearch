"""Problem 09: Transformer Decoder Layer with D-FINE Target Gating"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from attention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer with Target Gating mechanism."""
    
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.sa   = MultiHeadAttention(d_model, num_heads, dropout)
        self.ca   = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Target Gating Layer (TGL)
        self.gate1 = nn.Linear(2*d_model, d_model)
        self.gate2 = nn.Linear(2*d_model, d_model)
        # FFN
        self.ff1 = nn.Linear(d_model, d_ffn)
        self.ff2 = nn.Linear(d_ffn, d_model)
        self.act  = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_query_pos, memory_pos, memory_key_padding_mask=None):
        """Forward pass with target gating."""
        # 1. Pre-LN Self-Attention
        n1 = self.norm1(tgt)
        q_s = k_s = n1 + tgt_query_pos
        sa_out, _ = self.sa(q_s, k_s, n1)
        tgt = tgt + self.dropout(sa_out)

        # 2. Cross-Attention (no residual — replaced by gating)
        n2 = self.norm2(tgt)
        q_c = n2 + tgt_query_pos
        k_c = memory + memory_pos
        cross_out, _ = self.ca(q_c, k_c, memory, key_padding_mask=memory_key_padding_mask)

        # 3. Target Gating
        gate_in = torch.cat([tgt, cross_out], dim=-1)  # (B, N, 2d)
        g1 = torch.sigmoid(self.gate1(gate_in))
        g2 = torch.sigmoid(self.gate2(gate_in))
        tgt = g1 * tgt + g2 * cross_out

        # 4. FFN
        n3 = self.norm3(tgt)
        ffn_out = self.ff2(self.dropout(self.act(self.ff1(n3))))
        return tgt + self.dropout(ffn_out)
