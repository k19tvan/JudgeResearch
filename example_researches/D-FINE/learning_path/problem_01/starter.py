import torch

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.
    
    Args:
        x: Tensor of shape (..., 4) containing boxes in cxcywh format.
        
    Returns:
        Tensor of shape (..., 4) containing boxes in xyxy format.
    """
    raise NotImplementedError("Implement bounding box conversion.")

def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.
    
    Args:
        x: Tensor of shape (..., 4) containing boxes in xyxy format.
        
    Returns:
        Tensor of shape (..., 4) containing boxes in cxcywh format.
    """
    raise NotImplementedError("Implement bounding box conversion.")
