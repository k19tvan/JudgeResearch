# learning_path/problem_08/starter.py
import torch
import torch.nn.functional as F

def build_cost_matrix(pred_logits: torch.Tensor, pred_boxes: torch.Tensor, 
                      tgt_labels: torch.Tensor, tgt_boxes: torch.Tensor,
                      weight_cls=1.0, weight_l1=1.0, weight_giou=1.0) -> torch.Tensor:
    """
    Args:
        pred_logits: (N, Num_Classes)
        pred_boxes: (N, 4) in CXCYWH or XYXY based on choice
        tgt_labels: (M,)
        tgt_boxes: (M, 4)
    Returns:
        cost_matrix: (N, M)
    """
    raise NotImplementedError("Calculate pair-wise Cost Matrix")
