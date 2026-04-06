# Problem 03 - Focal Loss

## Description
Calculate binary focal loss given raw unnormalized logits.

## Input Format
`inputs`: float32 tensor of shape `(N, C)` (logits).
`targets`: float32 tensor of shape `(N, C)` (values `0.0` or `1.0`).

## Output Format
Loss tensor reduction applied depending on the `reduction` arg ('none' -> `(N, C)`).

## Constraints
`targets` must be exactly 0 or 1. No soft labels.

## Example
**Input:**
```python
x = torch.tensor([[0.5, 0.5, 1.0, 1.0]]) # For problem 01 cxcywh
```

**Output:**
```python
# Expected xyxy output
tensor([[0.0, 0.0, 1.0, 1.0]])
```

## Hints
- Check your broadcasting dimensions meticulously.
- Leverage `torch.max` / `.unbind` as required.
- Do not use `for` loops.

## Checker
Run the provided checker to validate your implementation:
`python checker.py`
