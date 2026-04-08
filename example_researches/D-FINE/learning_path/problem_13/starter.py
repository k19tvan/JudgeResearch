"""Problem 13: Starter Code - Matching Union for Multi-Layer Supervision"""
import torch
import torch.nn as nn
from typing import List, Tuple


class HungarianMatcher(nn.Module):
    """Hungarian Matcher with matching union computation."""
    
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        """Hungarian matching - implement in Problem 03."""
        # Simplified - students implement in P03
        raise NotImplementedError("Implement single-layer matching in Problem 03")
    
    def compute_matching_union(self, indices_list: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Compute matching union (consensus across layers).
        
        TODO: Implement this method according to specifications in question.md
        
        Args:
            indices_list: List of (src_indices, tgt_indices) from each layer
                         Each element: ([num_m at layer i],), ([num_m at layer i],)
        
        Returns:
            matching_union: List of (src_consensus, tgt_consensus) tuples
        """
        # PLACEHOLDER - TODO implement
        raise NotImplementedError("Implement compute_matching_union")
