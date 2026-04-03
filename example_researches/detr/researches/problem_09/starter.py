# learning_path/problem_09/starter.py
import torch
from scipy.optimize import linear_sum_assignment

class HungarianMatcher:
    def __init__(self):
        pass

    @torch.no_grad()
    def forward(self, cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cost_matrix: (N, M)
        Returns:
            Tuple of (pred_idx, tgt_idx) where both are (M,) sized 1D tensors
            representing the optimal matching assignment pairings.
        """
        raise NotImplementedError("Use scipy to solve linear sum assignment")
