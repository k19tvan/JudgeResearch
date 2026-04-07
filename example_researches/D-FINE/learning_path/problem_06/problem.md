# Problem 06 - 2D Sine Positional Encoding

## Description
Transformers are permutation-invariant — they have no inherent notion of spatial position. In DETR/D-FINE, the backbone produces a 2D feature map `(B, C, H, W)` which is flattened to `(B, H*W, C)`. To tell the transformer WHERE each patch is in the image, 2D sinusoidal positional encodings are added. Your task is to implement `PositionEmbeddingSine2D` which generates a `(B, H*W, d_model)` encoding tensor from a `(B, C, H, W)` feature map.

## Input Format
`x`: Tensor of shape `(B, C, H, W)`, float32 — feature map (C is backbone channels, not d_model).
`mask` (optional): Tensor of shape `(B, H, W)`, bool — True for padding positions. If None, assume no padding.

## Output Format
Tensor of shape `(B, H*W, d_model)`, float32 — sinusoidal encodings for each spatial position.

## Constraints
- `d_model` must be even (half for x, half for y).
- Encoding values must be in `[-1, 1]`.
- No learnable parameters.
- Must handle arbitrary `(H, W)` at inference time.

## Example
```python
pe = PositionEmbeddingSine2D(d_model=256, temperature=10000)
x  = torch.zeros(1, 512, 8, 8)   # B=1, C=512, H=8, W=8
enc = pe(x)                       # (1, 64, 256)
assert enc.shape == (1, 64, 256)
assert enc.abs().max() <= 1.0 + 1e-5
```

## Hints
- Create normalized grids `y_embed (H, W)` and `x_embed (H, W)` ranging in `[0, 1]`.
- Create dimension indices `dim_t` from 0 to `d_model//2 - 1`.
- Temperature formula: `W(i) = temperature^(2i/d_model)`.
- For x-axis: `sin(x/W(0)), cos(x/W(0)), sin(x/W(1)), cos(x/W(1)), ...`.
- Stack [sin, cos] pairs then reshape to `(d_model//2, 2)` → `(d_model,)`.
- Final result: concat x-encoding `(B, H, W, d_model//2)` and y-encoding `(B, H, W, d_model//2)` along last dim
  → `(B, H, W, d_model)` → flatten spatial → `(B, H*W, d_model)`.

## Checker
```bash
python checker.py
```
