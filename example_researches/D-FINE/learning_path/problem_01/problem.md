# Problem 01 - Bounding Box Format Conversion

## Description
Object detection models utilize varying bounding box topologies. DETR-style architectures natively predict boxes using a normalized `(cx, cy, w, h)` form (center coordinates, width, and height), but many loss functions and evaluations like Intersection over Union (IoU) operate strictly on Euclidean corners `(x1, y1, x2, y2)`. Your goal is to implement purely vectorized PyTorch tensor functions that transform bounding box coordinate streams bidirectionally. 

## Input Format
`boxes`: A PyTorch tensor of shape `(B, N, 4)` or `(N, 4)` containing float32 coordinates.
- If navigating `cxcywh` space, the last dimension is precisely `[center_x, center_y, width, height]`.
- If navigating `xyxy` space, the last dimension is strictly `[xmin, ymin, xmax, ymax]`.

## Output Format
A PyTorch tensor adhering to the exact same shape `(B, N, 4)` and `dtype` as the input, containing the mathematically transposed bounding boxes.

## Constraints
- Width (`w`) and height (`h`) should always logically remain non-negative parameters.
- Target `xmin <= xmax` and `ymin <= ymax` characteristics.
- Loops arrays, iterations, and non-vectorized abstractions risk severe GPU penalties. **No loop mechanisms are allowed.** All alterations must be done via pure math tensor operations.

## Example
**Input Tensor (`cxcywh`):**
```python
boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
```

**Output Tensor (`xyxy`):**
```python
converted = torch.tensor([[0.4, 0.3, 0.6, 0.7]])
```

## Hints
- `x1 = cx - w / 2`
- Utilize `torch.unbind(boxes, dim=-1)` which splits the inner-most dimension into 4 individual parameter vectors.
- Once mathematics hit each respective parameter, reformulate them synchronously via `torch.stack((p1, p2, p3, p4), dim=-1)`.

## Checker
Test the pipeline constraints via:
```bash
python checker.py
```
