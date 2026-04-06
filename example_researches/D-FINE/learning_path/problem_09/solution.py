import torch
import torch.nn as nn

class BoxRefinement(nn.Module):
    """
    Problem 09: Fine-Grained Box Refinement
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, features, reference_boxes):
        """
        features: [B, N, C]
        reference_boxes: [B, N, 4] (cxcywh)
        """
        offsets = self.regressor(features)
        refined_boxes = reference_boxes + offsets # In paper, it's inverse sigmoid mapping
        return torch.sigmoid(refined_boxes)
