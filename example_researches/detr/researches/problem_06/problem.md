<!-- learning_path/problem_06/problem.md -->
# Problem 06 - Transformer Decoder Layer

## Description
- The DETR Decoder takes learned "Object Queries" and extracts context from the Encoder.
- It involves two attention blocks:
  1. **Self-Attention** on object queries (lets objects "talk" to each other to avoid duplicate predictions).
  2. **Cross-Attention** on encoder outputs (lets objects query the image features).

### Data Specification and Shapes
- `N_obj`: Number of object queries (e.g., 100)
- `N_src`: Sequence length of encoder output (Image pixels)
- Input `tgt`: `(N_obj, B, C)`
- `memory`: Encoder output `(N_src, B, C)`
- `query_embed`: Object positional query `(N_obj, B, C)`
- Output: `(N_obj, B, C)`

## Requirements
- Self-attention: Queries/Keys combine `tgt` + `query_embed`. Values equal `tgt`.
- Cross-attention: Queries equal `tgt` + `query_embed`. Keys combine `memory` + `pos_embed`. Values equal `memory`.
- Apply appropriate FFN, LayerNorms, and residual connections.

## Theory
Cross-Attention operation logic:
$$ Q_{cross} = tgt + query\_embed $$
$$ K_{cross} = memory + pos\_embed $$
$$ V_{cross} = memory $$
This structural separation effectively decouples "what to look for" ($Q$) and "where it is" ($K$) from the raw visual semantic data ($V$).

## Checker
```bash
python learning_path/problem_06/checker.py
```
