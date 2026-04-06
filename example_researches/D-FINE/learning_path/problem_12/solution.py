import torch
from scipy.optimize import linear_sum_assignment

class HungarianMatcher:
    def __init__(self):
        pass

    def __call__(self, cost_matrix):
        """
        cost_matrix: [B, N, M]
        Returns list of (query_indices, target_indices) for each batch
        """
        indices = []
        for i, C in enumerate(cost_matrix.detach().cpu().numpy()):
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), 
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices
