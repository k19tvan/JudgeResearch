import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple


class HungarianMatcher(nn.Module):
    """
    Performs optimal bipartite matching between predictions and ground-truth boxes.
    Computes a cost matrix combining classification, L1 box, and GIoU costs.
    Uses scipy's linear_sum_assignment to find the optimal matching.
    
    Args:
        cost_class: Weight for classification cost (default: 2.0)
        cost_bbox: Weight for L1 bounding box cost (default: 5.0)
        cost_giou: Weight for GIoU cost (default: 2.0)
        alpha: Alpha parameter for focal cost (default: 0.25)
        gamma: Gamma parameter for focal cost (default: 2.0)
    """
    
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict:
        """
        Args:
            outputs: dict with 'pred_logits' (B, N, C) and 'pred_boxes' (B, N, 4)
            targets: list of B dicts, each with 'labels' (T_i,) and 'boxes' (T_i, 4)
        
        Returns:
            dict with 'indices': list of B tuples (src_idx, tgt_idx)
        """
        raise NotImplementedError
