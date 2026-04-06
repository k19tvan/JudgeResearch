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

## Theory Snapshot
- Hungarian matching solves one-to-one assignment globally, not greedily.
- It converts dense pairwise costs into optimal index pairs for supervision.
- Full optimization formulation is in [researches/problem_09/theory.md](researches/problem_09/theory.md).

## Checker
```bash
python learning_path/problem_09/checker.py
```
