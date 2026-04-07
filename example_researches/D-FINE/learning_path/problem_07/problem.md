# Problem 07 - Multi-Head Self-Attention

## Description
Multi-Head Attention (MHA) is the core computational building block of all transformer architectures. It allows each token to gather information from every other token using learned Query, Key, Value projections combined with scaled dot-product attention. In D-FINE, MHA is used in both the Hybrid Encoder (self-attention over feature maps) and the Transformer Decoder (self-attention over object queries and cross-attention to image features). Your task is to implement `MultiHeadAttention` from scratch using only `torch.nn.Linear` and tensor operations (no `nn.MultiheadAttention`).

## Input Format
`q`: `(B, T_q, d_model)` — query sequence.
`k`: `(B, T_k, d_model)` — key sequence.
`v`: `(B, T_k, d_model)` — value sequence.
`key_padding_mask` (optional): `(B, T_k)` bool — True for positions to be ignored.

## Output Format
`out`: `(B, T_q, d_model)` — attended output.
`attn_weights`: `(B, H, T_q, T_k)` — attention weight matrix (for visualization).

## Constraints
- `d_model` must be divisible by `num_heads`.
- Scaling by `1 / sqrt(d_k)` where `d_k = d_model // num_heads`.
- `key_padding_mask` adds `-inf` before softmax if provided.
- No `F.multi_head_attention_forward`.

## Example
```python
mha = MultiHeadAttention(d_model=256, num_heads=8, dropout=0.0)
q = torch.randn(2, 10, 256)   # (B, T_q, d_model)
k = v = torch.randn(2, 50, 256)
out, attn = mha(q, k, v)
assert out.shape == (2, 10, 256)
assert attn.shape == (2, 8, 10, 50)
```

## Hints
- Define `W_q, W_k, W_v, W_o` as `nn.Linear(d_model, d_model, bias=True)`.
- `d_k = d_model // num_heads`.
- Reshape Q,K,V from `(B, T, d_model)` → `(B, T, H, d_k)` → `(B, H, T, d_k)`.
- `scores = Q @ K.transpose(-2,-1) / sqrt(d_k)` → `(B, H, T_q, T_k)`.
- Mask: `scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))`.
- `attn = softmax(scores, dim=-1)`. Dropout on attn.
- `out = attn @ V` → `(B, H, T_q, d_k)` → reshape → `(B, T_q, d_model)` → `W_o`.

## Checker
```bash
python checker.py
```
