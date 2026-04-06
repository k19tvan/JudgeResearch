# Problem 15 - Component Implementation

## Description
Implement the sub-module correctly according to mathematical theory provided in D-FINE.

## Input Format
A PyTorch tensor or dict of tensors depending on the specific module.

## Output Format
A scaled/formatted target PyTorch tensor.

## Constraints
Strict dimension validation per the batch size config.

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
