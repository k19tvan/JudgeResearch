# learning_path/problem_11/starter.py
import torch
import torch.nn as nn

def train_one_step(model: nn.Module, criterion: nn.Module, matcher, optimizer: torch.optim.Optimizer, 
                   samples: torch.Tensor, targets: list[dict], max_norm: float = 0.1):
    """
    Args:
        model: DETR network predicting logits and boxes
        criterion: SetCriterion computing the bipartite loss
        matcher: HungarianMatcher giving optimal alignments
        optimizer: PyTorch optimizer
        samples: (B, 3, H, W)
        targets: list of B dicts with 'labels' and 'boxes'
        max_norm: gradient clipping value
    Returns:
        loss_val: python scalar float
    """
    raise NotImplementedError("Implement the completed backprop loop")
