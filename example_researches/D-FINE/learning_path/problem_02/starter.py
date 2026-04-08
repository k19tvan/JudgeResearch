import torch
from torch import Tensor


def box_area(boxes: Tensor) -> Tensor:
    """
    Compute area of boxes in xyxy format.
    
    Args:
        boxes: Tensor of shape (N, 4) where last dimension is [x1, y1, x2, y2]
    
    Returns:
        Tensor of shape (N,) containing areas
    """
    
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    raise NotImplementedError


def box_iou(boxes1: Tensor, boxes2: Tensor):
    """
    Compute pairwise Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format
    
    Returns:
        iou: Tensor of shape (N, M) containing IoU values
        union: Tensor of shape (N, M) containing union areas
    """
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # N x M x 2
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # N x M x 2

    wh = (rb - lt).clamp(min=0) # N x M x 2 (w, h)
    inter = wh[:, :, 0] * wh[:, :, 1] # N x M
    
    # N x 1 + M - N x M
    union = area1[:, None] + area2 - inter 
    iou = inter / union.clamp(min=1e-6) 
    
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute pairwise Generalized IoU (GIoU) between two sets of boxes.
    GIoU adds a penalty term for non-overlapping boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format
    
    Returns:
        Tensor of shape (N, M) containing GIoU values in [-1, 1]
    """
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1] 
    
    return iou - (area - union) / area.clamp(min=1e-6)
    


