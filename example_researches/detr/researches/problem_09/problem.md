<!-- learning_path/problem_09/problem.md -->
# Problem 09 - Hungarian Matcher Assignment

## Description
- Given our computed pairwise Cost Matrix (Problem 08), we must find the optimal 1-to-1 matching.
- Use `scipy.optimize.linear_sum_assignment` which acts on the $(N, M)$ matrix optimally minimizing total cost.

### Data Specification and Shapes
- Cost Matrix: `(N, M)` (N queries, M targets)
- Output: A tuple `(query_indices, target_indices)` indicating which query corresponds to which target.

## Requirements
- Take the detached CPU numpy version of the cost matrix.
- Pass it to the hungarian solver algorithm.
- Return the indices scaled back up to PyTorch tensors.

## Theory
Given cost matrix $C \in \mathbb{R}^{N \times M}$, find permutation matrix $P$ minimizing:
$$ \arg\min_{P} \sum_{i,j} C_{i,j} P_{i,j} $$
This solves exactly the problem of identifying the exact structural assignment without heuristics or hard IoU thresholds.

## Checker
```bash
python learning_path/problem_09/checker.py
```
