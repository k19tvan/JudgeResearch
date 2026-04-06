<!-- learning_path/problem_10/problem.md -->
# Problem 10 - Set Criterion Loss

## Description
- Having assigned predictions to targets, we now compute the loss to properly construct gradients.
- We implement Classification (Cross Entropy), L1 Box, and GIoU Box losses.
- Unlike Faster R-CNN, DETR considers negative "bg" classes actively in the classification loss for unmatched queries.

### Data Specification and Shapes
- Output is a scalar subset of `loss_dict` combining metrics.

## Requirements
- Gather the specifically assigned predictions using `pred_idx` and `tgt_idx`.
- Compute classification cross-entropy over ALL queries (unmatched queries belong to background class).
- Compute L1 and GIoU bounds ONLY for the positively matched queries.
- Return a dictionary of losses.

## Theory Snapshot
- Set criterion combines class loss over all queries and box losses over matched pairs.
- Background handling for unmatched queries is central to stable DETR training.
- Full loss decomposition is in [researches/problem_10/theory.md](researches/problem_10/theory.md).

## Checker
```bash
python learning_path/problem_10/checker.py
```
