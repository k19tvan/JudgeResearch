import torch
from typing import Tuple

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes.
    Args:
        boxes: Tensor of shape (N, 4) in xyxy format.
    Returns:
        Tensor of shape (N,) containing the areas.
    """
    raise NotImplementedError("Implement box_area")

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format
    Returns:
        Tuple of (iou, union) where both have shape (N, M).
    """
    raise NotImplementedError("Implement box_iou")

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Return generalized intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format
    Returns:
        Tensor of shape (N, M) containing pair-wise GIoU values.
    """
    raise NotImplementedError("Implement generalized_box_iou")