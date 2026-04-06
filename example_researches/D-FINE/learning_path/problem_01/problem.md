# Problem 01 - Bounding Box Coordinate Conversions

## Description
- In modern object detectors like D-FINE, bounding boxes are represented in multiple formats depending on the operation.
- The model outputs box centers and dimensions (CX, CY, W, H) as this is easier for neural networks to regress.
- However, calculating Intersection over Union (IoU) and drawing boxes requires the corners (X1, Y1, X2, Y2).
- Your goal is to write heavily vectorized PyTorch functions to convert between these two representations quickly, without using Python loops.

### Data Specification and Shapes
- `x`: A PyTorch Tensor containing coordinates.
- Dimension formats:
  - `cxcywh`: center_x, center_y, width, height
  - `xyxy`: top_left_x, top_left_y, bottom_right_x, bottom_right_y

## Requirements
- Implement `box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor`
- Implement `box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor`
- Operations must be fully vectorized (operate on the last dimension of the tensor).
- Do not use `for` loops.

## Hints
- Use `tensor.unbind(-1)` to split the tensor into 4 separate coordinate tensors.
- Compute the new coordinates for each variable individually.
- Use `torch.stack()` with `dim=-1` to recombine them into the final tensor before returning.

## Theory Snapshot
- `cx, cy` are the center coordinates of the box.
- `w, h` are the total width and height.
- `x1, y1` are the top-left coordinates: `x1 = cx - w/2`
- `x2, y2` are the bottom-right coordinates: `x2 = cx + w/2`

## Checker
Run the provided checker to validate your implementation:
`python checker.py`
