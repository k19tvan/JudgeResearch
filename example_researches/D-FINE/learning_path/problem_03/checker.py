import torch
from starter import sigmoid_focal_loss
import torch.nn.functional as F

def run_checks():
    inputs = torch.tensor([
        [-5.0, 5.0],   # Strong negative, Strong positive (EASY)
        [-0.1, 0.1]    # Weak negative, Weak positive (HARD)
    ], dtype=torch.float32)
    
    targets = torch.tensor([
        [0.0, 1.0], 
        [0.0, 1.0]
    ], dtype=torch.float32)

    loss = sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0)
    
    assert loss.shape == (2, 2), "Loss shape mismatch. Reduction must default to 'none'."

    # Check focal behavior: The 'HARD' example should have much larger loss than the 'EASY' one.
    assert loss[0, 1] < loss[1, 1], "Hard positive must have a higher loss than Easy positive."
    assert loss[0, 0] < loss[1, 0], "Hard negative must have a higher loss than Easy negative."
    
    print("All Problem 03 checks passed")

if __name__ == "__main__":
    run_checks()