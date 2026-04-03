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

## Theory
Total loss is defined as:
$$ L(y, \hat{y}) = \lambda_{cls} L_{CE} + \mathbb{1}_{\{\text{match}\}} \left[ \lambda_{\text{L1}} L_1 + \lambda_{\text{giou}} L_{giou} \right] $$

## Checker
```bash
python learning_path/problem_10/checker.py
```
