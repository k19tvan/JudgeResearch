import os
import re

base_dir = "/home/enn/workspace/project/AI_Judge/example_researches/D-FINE/learning_path"

problems = [
    {"id": "01", "name": "Bounding Box Coordinate Conversions", "desc": "Convert bounding box representations from (cx, cy, w, h) to (x1, y1, x2, y2) and vice versa.", "in": "A tensor `x` of shape `(..., 4)` containing boxes.", "out": "A tensor of shape `(..., 4)` containing converted boxes.", "const": "- Coordinates can be any real float32 value.\n- `w` and `h` are non-negative."},
    {"id": "02", "name": "Generalized IoU (GIoU)", "desc": "Compute Pairwise IoU and Generalized IoU for two sets of bounding boxes.", "in": "`boxes1`: float32 tensor of shape `(N, 4)` in `xyxy` format.\n`boxes2`: float32 tensor of shape `(M, 4)` in `xyxy` format.", "out": "`iou`: float32 tensor of shape `(N, M)`.\n`giou`: float32 tensor of shape `(N, M)`.", "const": "- `x2 >= x1` and `y2 >= y1` for all boxes.\n- Output ranges: IoU in `[0, 1]`, GIoU in `[-1, 1]`."},
    {"id": "03", "name": "Focal Loss", "desc": "Calculate binary focal loss given raw unnormalized logits.", "in": "`inputs`: float32 tensor of shape `(N, C)` (logits).\n`targets`: float32 tensor of shape `(N, C)` (values `0.0` or `1.0`).", "out": "Loss tensor reduction applied depending on the `reduction` arg ('none' -> `(N, C)`).", "const": "`targets` must be exactly 0 or 1. No soft labels."},
]

for i in range(4, 17):
    problems.append({
        "id": f"{i:02d}", 
        "name": "Component Implementation", 
        "desc": "Implement the sub-module correctly according to mathematical theory provided in D-FINE.",
        "in": "A PyTorch tensor or dict of tensors depending on the specific module.",
        "out": "A scaled/formatted target PyTorch tensor.",
        "const": "Strict dimension validation per the batch size config."
    })

for p in problems:
    folder = os.path.join(base_dir, f"problem_{p['id']}")
    if not os.path.exists(folder): continue
    
    filepath = os.path.join(folder, "problem.md")
    
    new_content = f"""# Problem {p['id']} - {p['name']}

## Description
{p['desc']}

## Input Format
{p['in']}

## Output Format
{p['out']}

## Constraints
{p['const']}

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
"""
    with open(filepath, "w") as f:
        f.write(new_content)

print("Rewritten problem.md files to CP style.")
