# learning_path/problem_05/starter.py
import torch
import torch.nn as nn
from learning_path.problem_04.starter import MultiHeadAttentionWithPos # Assuming availability

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithPos(d_model, nhead)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (N, B, C) Image feature patches
            pos_embed: (N, B, C) Spatial 2D encodings
        Returns:
            output: (N, B, C) Encoded patches
        """
        raise NotImplementedError("Implement the full Encoder Layer")
