import torch
import torch.nn as nn

class DecoupledHeads(nn.Module):
    """
    Problem 10: Decoupled Heads
    """
    def __init__(self, hidden_dim=256, num_classes=80):
        super().__init__()
        self.cls_branch = nn.Linear(hidden_dim, num_classes)
        self.reg_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, query_features):
        cls_logits = self.cls_branch(query_features)
        box_offsets = self.reg_branch(query_features)
        return cls_logits, box_offsets
