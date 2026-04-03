<!-- learning_path/problem_04/problem.md -->
# Problem 04 - Attention Mechanics

## Description
- Self and Cross Attention form the core of the Transformer.
- You will implement the Multi-Head Attention layer to process queries, keys, and values.
- Crucially, DETR adds Positional Encodings directly to Queries and Keys to retain spatial information without distorting the Value representation.

### Data Specification and Shapes
- `N`: Sequence length (pixels / objects)
- `B`: Batch size
- `C`: Embedding dimension
- Input `query`, `key`, `value`: `(N, B, C)`
- `pos_embed`: Positional encoding of shape `(N, B, C)`

## Requirements
- Support passing queries and keys with their specific positional embeddings added exactly before generating attention weights.
- Compute attention scores using scaled dot-product.
- Apply softmax and multiply by value embeddings.

## Theory
Given Query $Q$, Key $K$, Value $V$, and positional encodings $P_Q, P_K$:
$$ \text{score} = \text{softmax}\left(\frac{(Q + P_Q)(K + P_K)^T}{\sqrt{d_k}}\right) V $$
Notice that positional encodings do not corrupt the value tensor $V$.

## Checker
```bash
python learning_path/problem_04/checker.py
```
