# Problem 01 - Bounding Box Coordinate Conversions

## Description
Convert bounding box representations from (cx, cy, w, h) to (x1, y1, x2, y2) and vice versa.

## Input Format
A tensor `x` of shape `(..., 4)` containing boxes.

## Output Format
A tensor of shape `(..., 4)` containing converted boxes.

## Constraints
- Coordinates can be any real float32 value.
- `w` and `h` are non-negative.

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
