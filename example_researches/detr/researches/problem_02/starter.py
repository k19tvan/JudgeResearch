# learning_path/problem_02/starter.py
import torch

def intersect(box1: torch.tensor, box2: torch.tensor):

    x_mn_1, y_mn_1, x_mx_1, y_mx_1 = box1
    x_mn_2, y_mn_2, x_mx_2, y_mx_2 = box2
    
    if max(x_mn_1, x_mn_2) > min(x_mx_1, x_mx_2): return 0
    if max(y_mn_2, y_mn_2) > min(y_mx_2, y_mx_2): return 0
    
    w = min(x_mx_1, x_mx_2) - max(x_mn_1, x_mn_2)
    h = min(y_mx_2, y_mx_2) - max(y_mn_2, y_mn_2)
    
    return w * h
    
def union(box1: torch.tensor, box2: torch.tensor):
    x_mn_1, y_mn_1, x_mx_1, y_mx_1 = box1
    x_mn_2, y_mn_2, x_mx_2, y_mx_2 = box2
    return (x_mx_1 - x_mn_1) * (y_mx_1 - y_mn_1) + (x_mx_2 - x_mn_2) * (y_mx_2 - y_mn_2)
    
def cal_c(box1: torch.tensor, box2: torch.tensor):
    x_mn_1, y_mn_1, x_mx_1, y_mx_1 = box1
    x_mn_2, y_mn_2, x_mx_2, y_mx_2 = box2
    
    x_mn = min(x_mn_1, x_mn_2)
    x_mx = max(x_mx_1, x_mx_2)
    y_mn = min(y_mn_1, y_mn_2)
    y_mx = max(y_mx_1, y_mx_2)
    
    return (x_mx - x_mn) * (y_mx - y_mn)
    
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format
    Returns:
        iou: (N, M) matrix
        union: (N, M) matrix
    """

    intersects = torch.tensor([[intersect(box1, box2) for box1 in boxes1] for box2 in boxes2])
    unions = torch.tensor([[union(box1, box2) for box1 in boxes1] for box2 in boxes2])
    
    return (intersects / unions, unions)

def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format
    Returns:
    
        giou: (N, M) matrix
    """
    
    cs =  torch.tensor([[cal_c(box1, box2) for box1 in boxes1] for box2 in boxes2])
    IoU, unions = box_iou(boxes1, boxes2)
    
    return IoU - (cs - unions) / cs
    
    
    raise NotImplementedError("Implement Generalized IoU")
