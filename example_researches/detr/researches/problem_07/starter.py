# learning_path/problem_07/starter.py
import torch
import torch.nn as nn

class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, hidden_dim):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        self.backbone = backbone
        self.transformer = transformer
        
        # 1. Define query embeddings
        # 2. Define class embed head (Linear layer)
        # 3. Define bbox embed head (3-layer MLP)
        
        raise NotImplementedError("Initialize DETR heads")

    def forward(self, samples: torch.Tensor):
        """
        Args:
            samples: (B, 3, H, W)
        Returns:
            dict containing:
            - 'pred_logits': (B, N, Num_Classes + 1)
            - 'pred_boxes': (B, N, 4)
        """
        raise NotImplementedError("Implement the DETR forward pass")
