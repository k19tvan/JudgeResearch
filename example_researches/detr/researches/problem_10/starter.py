# learning_path/problem_10/starter.py
import torch
import torch.nn.functional as F

class SetCriterion(torch.nn.Module):
    def __init__(self, num_classes, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        
    def forward(self, outputs, targets, indices):
        """
        Args:
            outputs: dict with 'pred_logits' and 'pred_boxes'
            targets: list of dicts with 'labels' and 'boxes'
            indices: Tuple of (pred_indices, target_indices) from Hungarian Matcher
        Returns:
            losses: Dict containing 'loss_ce', 'loss_bbox', 'loss_giou'
        """
        raise NotImplementedError("Implement Loss processing for bounded sets")
