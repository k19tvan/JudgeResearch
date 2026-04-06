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

## Theory Snapshot
- DETR injects position into query and key while keeping value as pure semantic content.
- This preserves geometric awareness in attention weights without distorting carried features.
- Full derivation and tensor-level interpretation are in [researches/problem_04/theory.md](researches/problem_04/theory.md).

## Checker
```bash
python learning_path/problem_04/checker.py
```
