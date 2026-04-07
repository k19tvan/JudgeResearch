# Problem 08 - Transformer Encoder Layer

## Description
A single Transformer Encoder Layer consists of two sub-layers: (1) Multi-Head Self-Attention over the sequence, (2) a Position-wise Feed-Forward Network (FFN). Each sub-layer is wrapped with a residual connection and Layer Normalization (Pre-LN or Post-LN). In D-FINE's Hybrid Encoder, multiple encoder layers are stacked to refine the flattened multi-scale feature maps before passing them to the decoder. Your task is to implement `TransformerEncoderLayer` accepting positional encodings as an additive input to queries and keys.

## Input Format
`src`: `(B, T, d_model)` — input token sequence (flattened feature map).
`pos`: `(B, T, d_model)` — positional encodings (from Problem 06), added to q and k.
`src_key_padding_mask` (optional): `(B, T)` bool — True for padded positions.

## Output Format
`out`: `(B, T, d_model)` — encoded sequence with same shape as input.

## Constraints
- Use pre-norm style: `LayerNorm → attention → residual`.
- Add `pos` to both q and k, NOT to v.
- FFN: `d_model → d_ffn → d_model` with GELU activation.
- Dropout applied after attention and after FFN.

## Example
```python
layer = TransformerEncoderLayer(d_model=256, num_heads=8, d_ffn=1024, dropout=0.0)
src = torch.randn(2, 49, 256)   # B=2, H*W=49 (7x7), d_model=256
pos = torch.randn(2, 49, 256)   # from PositionEmbeddingSine2D
out = layer(src, pos)
assert out.shape == (2, 49, 256)
```

## Hints
- In pre-LayerNorm: apply LN to `src`, then add `pos` to the normalized output for q and k.
- q, k, v = (norm_src + pos), (norm_src + pos), norm_src for self-attention.
- After attention output: `src = src + dropout(attn_out)` (residual).
- FFN: `nn.Linear(d_model, d_ffn)` → GELU → Dropout → `nn.Linear(d_ffn, d_model)`.
- After FFN: `src = src + dropout(ffn_out)` (second residual).

## Checker
```bash
python checker.py
```
