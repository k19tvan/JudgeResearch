<!-- learning_path/problem_01/problem.md -->
# Problem 01 - Box Format Conversions

## Description
- DETR processes bounding boxes in multiple formats.
- Dataloaders often provide `[x_min, y_min, w, h]` (COCO absolute format).
- The model outputs and losses expect normalized `[c_x, c_y, w, h]` format (center X, center Y, width, height) relative to image size [0, 1].
- Implement functions to convert between `xywh` and `cxcxw` formats.

### Data Specification and Shapes
- `B`: Batch size / Number of boxes
- Img Size: `[Height, Width]`
- Inputs/Outputs: `(B, 4)`

## Requirements
- `box_xywh_to_cxcxw`: calculate center coordinates and normalize by image dimensions.
- `box_cxcxw_to_xyxy`: given normalized format, calculate absolute `[x_min, y_min, x_max, y_max]`.

## Theory
Normalized center X is calculated as:
$$ c_x = \frac{x_{min} + w / 2}{W_{img}} $$
To revert back to absolute bounds:
$$ x_{min} = (c_x - w / 2) \cdot W_{img} $$

## Checker
```bash
python learning_path/problem_01/checker.py
```
