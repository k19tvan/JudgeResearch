# learning_path/problem_08/checker.py
import torch
from starter import build_cost_matrix

def check_cost_matrix():
    N, M, Num_Classes = 10, 3, 5
    
    pred_logits = torch.randn(N, Num_Classes)
    pred_boxes = torch.rand(N, 4)
    
    tgt_labels = torch.tensor([1, 4, 2])
    tgt_boxes = torch.rand(M, 4)
    
    try:
        cost_matrix = build_cost_matrix(pred_logits, pred_boxes, tgt_labels, tgt_boxes)
        assert cost_matrix.shape == (N, M), f"Matrix shape mismatch: {cost_matrix.shape}"
        assert not torch.isnan(cost_matrix).any(), "NaN found in costs"
        print("All Problem 08 checks passed")
    except NotImplementedError:
        print("Implement the Bipartite Cost Matrix.")
        raise

if __name__ == "__main__":
    check_cost_matrix()
