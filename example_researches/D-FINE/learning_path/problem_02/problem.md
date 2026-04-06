# Problem 02 - Generalized IoU (GIoU)

## Description
Compute Pairwise IoU and Generalized IoU for two sets of bounding boxes.

## Input Format
`boxes1`: float32 tensor of shape `(N, 4)` in `xyxy` format.
`boxes2`: float32 tensor of shape `(M, 4)` in `xyxy` format.

## Output Format
`iou`: float32 tensor of shape `(N, M)`.
`giou`: float32 tensor of shape `(N, M)`.

## Constraints
- `x2 >= x1` and `y2 >= y1` for all boxes.
- Output ranges: IoU in `[0, 1]`, GIoU in `[-1, 1]`.

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
