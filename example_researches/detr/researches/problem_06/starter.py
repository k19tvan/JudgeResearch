# learning_path/problem_06/starter.py
import torch
import torch.nn as nn
# from learning_path.problem_04.starter import MultiHeadAttentionWithPos 

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # self.self_attn = ...
        # self.multihead_attn = ...
        
        # FFN Layers
        # self.linear1 = ...
        # self.dropout = ...
        # self.linear2 = ...
        
        # ... Norms/Dropouts ...
        pass

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                query_embed: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: (N_obj, B, C)
            memory: (N_src, B, C)
            query_embed: (N_obj, B, C)
            pos_embed: (N_src, B, C)
        Returns:
            output: (N_obj, B, C) Decoder output
        """
        raise NotImplementedError("Implement the full Decoder Layer")
