# Problem 02 - Generalized IoU (GIoU)

## Description
- Intersection over Union (IoU) measures the overlap between two bounding boxes.
- However, if two boxes do not overlap at all, IoU is 0, providing no gradient for the network to pull them closer.
- Generalized IoU (GIoU) solves this by finding the smallest enclosing box covering both, and using the empty area of this container as a penalty term.
- Your task is to implement standard IoU and then GIoU, which will be the core metric of our bounding box loss.

### Data Specification and Shapes
- `boxes1`: Tensor of shape `(N, 4)` in `xyxy` format.
- `boxes2`: Tensor of shape `(M, 4)` in `xyxy` format.
- Outputs `iou` and `giou` must be tensors of shape `(N, M)`, representing pairwise comparisons.

## Requirements
- Implement `box_area(boxes: torch.Tensor) -> torch.Tensor`
- Implement `box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]` (returns iou, union)
- Implement `generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor`
- Do not use `for` loops. Must be fully vectorized using PyTorch broadcasting.

## Hints
- `unsqueeze()` can help trigger broadcasting to get `(N, M, 4)` shapes for pairwise intersections.
- Intersection corners can be found using `torch.max` for top-left and `torch.min` for bottom-right.
- Clamp the intersection width and height to a minimum of `0.0`.
- The enclosing box corners are found using `torch.min` for top-left and `torch.max` for bottom-right.

## Checker
Run the provided checker to validate your implementation:
`python checker.py`