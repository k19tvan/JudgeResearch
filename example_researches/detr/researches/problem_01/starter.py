# learning_path/problem_01/starter.py
import torch

def box_xywh_to_cxcxw(boxes: torch.Tensor, image_size: tuple) -> torch.Tensor:
    """
    Args:
        boxes: (N, 4) in absolute [x_min, y_min, w, h] format
        image_size: Tuple of (Height, Width)
    Returns:
        (N, 4) in normalized [center_x, center_y, w, h] format
    """
    
    x_min, y_min, w, h = boxes[0]
    h0, w0 = image_size
    
    return torch.tensor([[(x_min + w/2) / w0, (y_min + h/2) / h0, h / h0, w / w0]])

def box_cxcxw_to_xyxy(boxes: torch.Tensor, image_size: tuple) -> torch.Tensor:
    """
    Args:
        boxes: (N, 4) in normalized [center_x, center_y, w, h] format
        image_size: Tuple of (Height, Width)
    Returns:
        (N, 4) in absolute [x_min, y_min, x_max, y_max] format
    """
    
    c_x, c_y, w, h = boxes[0]
    h0, w0 = image_size
    
    c_x = c_x * w0
    c_y = c_y * h0
    
    w = w * w0
    h = h * h0
    
    x_min = c_x - w / 2 
    y_min = c_y - h / 2
    
    x_max = c_x + w / 2
    y_max = c_y + h / 2
    
    return torch.tensor([[x_min, y_min, x_max, y_max]])
