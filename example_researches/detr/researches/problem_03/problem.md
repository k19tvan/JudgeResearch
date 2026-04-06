<!-- learning_path/problem_03/problem.md -->
# Problem 03 - 2D Sine Positional Encoding

## Description
- The Transformer fundamentally lacks the concept of order or geometry.
- We must provide fixed 2D positional embeddings for the Image Feature map.
- The embedding is the concatenation of sine and cosine frequencies across the width and height dimensions.

### Data Specification and Shapes
- `B`: Batch size
- `C`: Embedding dimension. Must be an even number.
- `H`, `W`: Height and width of the map.
- Image Mask: `(B, H, W)` of bool indicating if a pixel is padded (True).
- Pos Encoding Output: `(B, C, H, W)`.

## Requirements
- Create unnormalized coordinates for rows and columns.
- Normalize them by max sequence length limits.
- Compute the frequencies and apply `sin` to even indices and `cos` to odd.
- Concatenate row and col embeddings to form total dimension `C`.

## Theory Snapshot
- 2D sine/cosine encoding injects spatial coordinates into attention without learnable position parameters.
- Separate embeddings for x and y axes are concatenated to build a full channel-wise positional vector.
- Full formulas, axis semantics, and shape flow are in [researches/problem_03/theory.md](researches/problem_03/theory.md).

## Checker
```bash
python learning_path/problem_03/checker.py
```
