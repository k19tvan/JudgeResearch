import json
import sys
import torch

def compute_giou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the GIoU matrix between N predicted boxes and M ground truth boxes.

    Parameters:
    pred_boxes (torch.Tensor): A tensor of shape (N, 4) representing predicted boxes.
                               Each box is [x_min, y_min, x_max, y_max].
    gt_boxes (torch.Tensor):   A tensor of shape (M, 4) representing ground truth boxes.
                               Each box is [x_min, y_min, x_max, y_max].

    Returns:
    torch.Tensor: An N x M tensor where the element at row i and column j is the 
                  GIoU value between pred_boxes[i] and gt_boxes[j].
    """
    pass

if __name__ == "__main__":
    with open("input.json", "r") as f:
        input_data = json.load(f)
    
    pred_boxes = torch.tensor(input_data.get("pred_boxes", []), dtype=torch.float32)
    gt_boxes = torch.tensor(input_data.get("gt_boxes", []), dtype=torch.float32)

    result_tensor = compute_giou_matrix(pred_boxes, gt_boxes)
    
    result = []
    if isinstance(result_tensor, torch.Tensor):
        result = result_tensor.tolist()
        
    with open("output.json", "w") as f:
        json.dump({"giou_matrix": result}, f)