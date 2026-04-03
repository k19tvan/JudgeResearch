# learning_path/problem_09/checker.py
import torch
from starter import HungarianMatcher

def check_hungarian():
    matcher = HungarianMatcher()
    
    # Simple explicit test case minimizing diagonal
    # N = 3 queries, M = 2 targets
    cost_matrix = torch.tensor([
        [0.1, 10.0],
        [10.0, 0.2],
        [5.0,  5.0]
    ])
    
    try:
        q_idx, t_idx = matcher.forward(cost_matrix)
        
        # Expected mapping: query 0 -> target 0. Query 1 -> target 1.
        assert torch.equal(q_idx, torch.tensor([0, 1])), f"Expected [0, 1] queries, got {q_idx}"
        assert torch.equal(t_idx, torch.tensor([0, 1])), f"Expected [0, 1] targets, got {t_idx}"
        
        print("All Problem 09 checks passed")
    except NotImplementedError:
        print("Implement Hungarian resolution logic")
        raise

if __name__ == "__main__":
    check_hungarian()
