<!-- learning_path/problem_02/problem.md -->
# Problem 02 - Generalized IoU (GIoU)

## Description
- Standard IoU fails to provide a meaningful gradient when two bounding boxes do not overlap (IoU = 0).
- DETR relies on GIoU to evaluate bounding box predictions, punishing boxes that differ greatly in scale or position regardless of non-overlap.
- Your task is to calculate the IoU and then the GIoU using absolute `[x_min, y_min, x_max, y_max]` coordinates.

### Data Specification and Shapes
- Input Boxes: `(N, 4)` and `(M, 4)` in `xyxy` format.
- Output: `(N, M)` matrix representing pairwise GIoU metric.

## Requirements
- Compute standard Intersection area.
- Compute the Union area.
- Find the area of the smallest enclosing box $C$ encompassing both targets.

## Theory
Given two absolute boxes $A$ and $B$:
$$ \text{IoU} = \frac{|A \cap B|}{|A \cup B|} $$
Let $C$ be the smallest convex bounding box containing $A$ and $B$.
$$ \text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|} $$

## Checker
```bash
python learning_path/problem_02/checker.py
```
