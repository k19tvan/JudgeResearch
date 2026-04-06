<!-- learning_path/problem_05/problem.md -->
# Problem 05 - Transformer Encoder Layer

## Description
- The DETR Encoder is a standard Transformer encoder layer.
- It applies Multi-Head Self-Attention followed by a Feed Forward Network (FFN).
- You will compose the Attention block you wrote with Layer Normalization and Dropout.

### Data Specification and Shapes
- `N`: Sequence length (H * W)
- `B`: Batch size
- `C`: Embedding dimension
- Input `src`: `(N, B, C)`
- `pos_embed`: `(N, B, C)`
- Output: `(N, B, C)`

## Requirements
- Add spatial positional encodings to `src` to form Queries and Keys. (Values remain exactly as `src`).
- Apply residual connection around the attention sub-layer: `src = src + Dropout(Attn())`
- Pass the result through the FFN with another residual connection.

## Theory Snapshot
- An encoder layer alternates global self-attention and channel-mixing FFN with residual paths.
- Positional encoding is added to Q/K to maintain spatial context across flattened image tokens.
- Full equations, normalization variants, and shape tracing are in [researches/problem_05/theory.md](researches/problem_05/theory.md).

## Checker
```bash
python learning_path/problem_05/checker.py
```
