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

## Theory
Positional encoding along axis $p$:
$$PE_{p, 2i} = \sin\left(\frac{p}{\tau^{2i / (C/2)}}\right), \quad PE_{p, 2i+1} = \cos\left(\frac{p}{\tau^{2i / (C/2)}}\right)$$
Here $p \in [0, 1]$ when normalized. Two such embeddings for $X$ ($C/2$ dim) and $Y$ ($C/2$ dim) are concatenated to create the $C$-dimensional vector.

## Checker
```bash
python learning_path/problem_03/checker.py
```
