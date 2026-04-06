import torch
import torch.nn as nn

class SimplifiedCrossAttention(nn.Module):
    """
    Problem 08: Decoder Cross Attention
    """
    def __init__(self, hidden_dim=256, nhead=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        # query: [B, N, C], key/value: [B, S, C]
        attn_out, _ = self.multihead_attn(query, key, value)
        return self.norm(query + attn_out)
