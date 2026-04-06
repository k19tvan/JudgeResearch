<!-- learning_path/problem_08/problem.md -->
# Problem 08 - Bipartite Cost Matrix

## Description
- Bipartite matching replaces traditional NMS logic.
- We must compute the distance (cost) between EVERY possible query prediction and EVERY target.
- Summing Classification, L1, and GIoU metrics produces this cost.

### Data Specification and Shapes
- `N`: Number of Queries
- `M`: Number of target objects
- `Num_Classes`: Classes + 1
- `pred_logits`: `(N, Num_Classes)`
- `pred_boxes`: `(N, 4)`
- Output: `(N, M)`

## Requirements
- Produce softly-scaled class probabilities using Softmax.
- Calculate L1 distance (pairwise) between pred boxes and target boxes using `cdist`.
- Use the `box_giou` (Problem 02) function or a built-in replacement to calculate shape penalties.
- Combine the matrices according to lambda weights.

## Theory Snapshot
- Matching cost fuses class evidence and geometric distance for each query-target pair.
- This cost matrix is the objective surface used by Hungarian assignment.
- Full weighted-cost derivation is in [researches/problem_08/theory.md](researches/problem_08/theory.md).

## Checker
```bash
python learning_path/problem_08/checker.py
```
