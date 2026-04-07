# Problem 09 - Transformer Decoder Layer with D-FINE Target Gating

## Description
D-FINE's Decoder Layer is an enhanced DETR Decoder. Each layer has three sub-layers: (1) Self-Attention over object queries, (2) Cross-Attention from object queries to image features (encoder output), and (3) an FFN. D-FINE introduces a **Target Gating Layer** after cross-attention that replaces the standard residual connection — it uses learned sigmoid gates to let each query dynamically choose how much to retain from its previous state vs. the new cross-attended context. Your task is to implement this full decoder layer.

## Input Format
`tgt`: `(B, N, d_model)` — object queries (N=num_queries, e.g. 300).
`memory`: `(B, T, d_model)` — encoder output (T=H*W spatial tokens).
`tgt_query_pos`: `(B, N, d_model)` — learnable query positional anchors.
`memory_pos`: `(B, T, d_model)` — 2D sinusoidal PE for memory.

## Output Format
`tgt_out`: `(B, N, d_model)` — updated object queries.

## Constraints
- Self-attention: q=k=tgt+pos, v=tgt (standard pre-LN).
- Cross-attention: q=(tgt+pos), k=(memory+memory_pos), v=memory.
- Target Gating (replaces cross-attn residual):
  `gate1 = sigmoid([tgt, cross_out] @ W1 + b1)`, `gate2 = sigmoid([tgt, cross_out] @ W2 + b2)`
  `tgt = gate1 * tgt + gate2 * cross_out`
- Then apply Pre-LN FFN as in Problem 08.

## Example
```python
layer = TransformerDecoderLayer(d_model=256, num_heads=8, d_ffn=1024)
tgt    = torch.randn(2, 300, 256)
memory = torch.randn(2, 400, 256)
qpos   = torch.randn(2, 300, 256)
mpos   = torch.randn(2, 400, 256)
out    = layer(tgt, memory, qpos, mpos)
assert out.shape == (2, 300, 256)
```

## Hints
- Concatenate tgt and cross_out along last dim for gating: `torch.cat([tgt, cross_out], dim=-1)` → `(B, N, 2*d_model)`.
- Gate linear layers: `nn.Linear(2*d_model, d_model)` for each gate.
- The gate replaces `tgt = tgt + cross_out` (NO residual add after cross-attn, only the gated combination).
- After gating: apply `LayerNorm` then `FFN + residual` as normal.

## Checker
```bash
python checker.py
```
