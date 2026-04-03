<!-- learning_path/problem_07/problem.md -->
# Problem 07 - Full DETR Assembly

## Description
- You will construct the core backbone + transformer architecture.
- Instead of using a single layer, you mock stacking encoders and decoders.
- You must declare the explicit MLP prediction heads for bounding boxes and classes.

### Data Specification and Shapes
- `B`: Batch size
- `C`: Hidden dimension size (e.g. 256)
- `N`: Number of object queries
- Input Image: `(B, 3, H, W)`
- Output Class Logits: `(B, N, Num_Classes + 1)`
- Output Bounding Boxes: `(B, N, 4)`

## Requirements
- Feed input to the backbone.
- Flatten the output spatial dimensions and add positional embeddings.
- Pass to the Transformer.
- Put the decoder sequence output through `class_embed` and `bbox_embed` logic.

## Theory
The ultimate pipeline matches the equation:
$$ Y = \text{MLP}(\text{Transformer}(\text{Backbone}(X), \text{Queries})) $$
where $Y$ includes both $Y_{cls}$ and $Y_{bbox}$.

## Checker
```bash
python learning_path/problem_07/checker.py
```
