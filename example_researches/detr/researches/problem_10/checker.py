# learning_path/problem_10/checker.py
import torch
from starter import SetCriterion

def check_criterion():
    num_classes = 5
    weights = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    criterion = SetCriterion(num_classes, weights)
    
    outputs = {
        'pred_logits': torch.randn(1, 10, num_classes + 1),
        'pred_boxes': torch.rand(1, 10, 4)
    }
    targets = [{'labels': torch.tensor([1, 3]), 'boxes': torch.rand(2, 4)}]
    indices = [(torch.tensor([0, 5]), torch.tensor([0, 1]))] # Mapped 0->0, 5->1
    
    try:
        losses = criterion(outputs, targets, indices)
        assert 'loss_ce' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        print("All Problem 10 checks passed")
    except NotImplementedError:
        print("Implement the Criterion Loss")
        raise

if __name__ == "__main__":
    check_criterion()
