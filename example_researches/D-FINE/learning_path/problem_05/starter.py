import torch
import torch.nn as nn
import torch.nn.functional as F


class SetCriterion(nn.Module):
    """
    Computes losses for D-FINE: VFL classification loss, L1 box regression, and GIoU regression.
    Requires matched indices from HungarianMatcher.
    
    Args:
        num_classes: Number of object classes
        weight_vfl: Weight coefficient for VFL loss (default: 1.0)
        weight_bbox: Weight coefficient for L1 box loss (default: 5.0)
        weight_giou: Weight coefficient for GIoU loss (default: 2.0)
    """
    
    def __init__(self, num_classes, weight_vfl=1.0, weight_bbox=5.0, weight_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.weight_vfl = weight_vfl
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

    def forward(self, outputs, targets, indices):
        """
        Args:
            outputs: dict with 'pred_logits' (B, N, C) and 'pred_boxes' (B, N, 4) in cxcywh
            targets: list of B dicts with 'labels' (T_i,) and 'boxes' (T_i, 4) in cxcywh
            indices: list of B tuples (src_idx, tgt_idx) from HungarianMatcher
        
        Returns:
            dict with 'loss_vfl', 'loss_bbox', 'loss_giou' (all normalized by num_boxes)
        """
        raise NotImplementedError
