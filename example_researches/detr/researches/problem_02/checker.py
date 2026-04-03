# learning_path/problem_02/checker.py
import torch
from starter import box_giou

def check_giou():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[10.0, 10.0, 20.0, 20.0]]) # No overlap
    
    giou = box_giou(b1, b2)
    assert giou.shape == (1, 1), f"Shape mismatch: {giou.shape}"
    
    # Area C = 20*20 = 400
    # Area Union = 100 + 100 = 200
    # IoU = 0
    # GIoU = 0 - (400 - 200)/400 = -0.5
    expected = torch.tensor([[-0.5]])
    assert torch.allclose(giou, expected), f"GIoU Math error: Expected -0.5, got {giou}"
    
    print("All Problem 02 checks passed")

if __name__ == "__main__":
    check_giou()
