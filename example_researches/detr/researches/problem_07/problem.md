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

## Theory Snapshot
- Full DETR composes backbone features, positional encoding, transformer processing, and prediction heads.
- Query slots produce parallel set predictions for class logits and normalized boxes.
- Full end-to-end formulation is in [researches/problem_07/theory.md](researches/problem_07/theory.md).

## Checker
```bash
python learning_path/problem_07/checker.py
```
