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

## Theory
Given an input $src$, the single encoder layer operates as follows:
$$ src' = src + \text{Dropout}(\text{MultiHeadAttention}(Q=src+P_K, K=src+P_K, V=src)) $$
$$ \text{output} = \text{LayerNorm}_1(src') + \text{Dropout}(\text{FFN}(\text{LayerNorm}_1(src'))) $$
*(Note: DETR uses slightly different layernorm positions in its default 'post' vs 'pre' config. Here we use standard Post-Normalization).*

## Checker
```bash
python learning_path/problem_05/checker.py
```
