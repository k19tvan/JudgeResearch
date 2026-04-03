# learning_path/problem_02/starter.py
import torch

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format
    Returns:
        iou: (N, M) matrix
        union: (N, M) matrix
    """
    raise NotImplementedError("Implement IoU")

def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format
    Returns:
        giou: (N, M) matrix
    """
    raise NotImplementedError("Implement Generalized IoU")
