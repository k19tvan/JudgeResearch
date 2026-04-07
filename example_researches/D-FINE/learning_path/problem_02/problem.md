# Problem 02 - Generalized Intersection over Union (GIoU)

## Description
Intersection over Union (IoU) measures overlap between two boxes but fails when boxes do not overlap at all — it returns 0 regardless of how far apart they are. Generalized IoU (GIoU) adds a penalty term involving the smallest enclosing box to provide a gradient signal even when boxes are non-overlapping. In D-FINE, GIoU is used both as a loss function component and as a cost signal inside the Hungarian Matcher. Your task is to implement `box_iou` and `generalized_box_iou` as fully vectorized PyTorch operations returning pairwise matrices.

## Input Format
`boxes1`: Tensor of shape `(N, 4)`, dtype `float32`, boxes in `[x1, y1, x2, y2]` format.
`boxes2`: Tensor of shape `(M, 4)`, dtype `float32`, boxes in `[x1, y1, x2, y2]` format.

Both tensors satisfy: `x2 >= x1` and `y2 >= y1`.

## Output Format
`box_iou(boxes1, boxes2)`:
- `iou`: Tensor of shape `(N, M)`, float32, values in `[0, 1]`.
- `union`: Tensor of shape `(N, M)`, float32, positive areas.

`generalized_box_iou(boxes1, boxes2)`:
- Returns Tensor of shape `(N, M)`, float32, values in `[-1, 1]`.

## Constraints
- All coordinates are float32 in `[0, 1]` (normalized).
- No Python loops. All operations must be vectorized.
- Degenerate boxes (zero-area) are valid inputs; handle them gracefully (clamp to min=0).
- GIoU is always `<= IoU` and `>= -1`.

## Example
```python
boxes1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])   # area=0.25
boxes2 = torch.tensor([[0.25, 0.25, 0.75, 0.75]])  # area=0.25
# Intersection: [0.25,0.25,0.5,0.5] -> area=0.0625
# Union = 0.25+0.25-0.0625 = 0.4375
# IoU = 0.0625/0.4375 ≈ 0.1429
# Enclosing: [0.0,0.0,0.75,0.75] -> area=0.5625
# GIoU = 0.1429 - (0.5625-0.4375)/0.5625 ≈ -0.0794
```

## Hints
- Use `torch.max(boxes1[:, None, :2], boxes2[:, :2])` for broadcasting intersection top-left corner.
- Use `(rb - lt).clamp(min=0)` to safely compute intersection width/height.
- For GIoU enclosing box: use `torch.min` for top-left and `torch.max` for bottom-right.
- `box_area(b)` equals `(b[:,2]-b[:,0]) * (b[:,3]-b[:,1])`.

## Checker
```bash
python checker.py
```
