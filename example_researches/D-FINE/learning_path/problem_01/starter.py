import torch

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from center-based (cx, cy, w, h) to corner-based (x1, y1, x2, y2) format.
    
    Args:
        x: Tensor of shape (..., 4) where last dimension is [cx, cy, w, h]
    
    Returns:
        Tensor of shape (..., 4) where last dimension is [x1, y1, x2, y2]
    """
    
    cx, cy, w, h = x.unbind(dim=-1)
    b = [(cx - w * 0.5), (cy - h * 0.5), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim = -1)
    
    raise NotImplementedError

def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from corner-based (x1, y1, x2, y2) to center-based (cx, cy, w, h) format.
    
    Args:
        x: Tensor of shape (..., 4) where last dimension is [x1, y1, x2, y2]
    
    Returns:
        Tensor of shape (..., 4) where last dimension is [cx, cy, w, h]
    """
    
    x1, y1, x2, y2 = x.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    
    b = [(x1 + w * 0.5), (y1 + h * 0.5), w, h]
    return torch.stack(b, dim=-1)
    
    raise NotImplementedError
