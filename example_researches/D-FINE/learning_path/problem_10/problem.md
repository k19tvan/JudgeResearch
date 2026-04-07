# Problem 10 - HGNetV2-S Stem and Depthwise Conv Stage

## Description
HGNetV2 (High-Performance GPU Net v2) is D-FINE's default backbone. Its design philosophy is to maximize accuracy-latency tradeoff on modern GPUs. The HGNetV2 Stem consists of: (1) an initial 3×3 conv + BN + ReLU block that downsamples the input from `(B, 3, H, W)` to `(B, 48, H/4, W/4)`, followed by (2) the first High-Resolution Global (LCS) block that applies Depthwise 3×3 convolutions for efficient local feature mixing. Your task is to implement this Stem + one LCS Block.

## Input Format
`x`: `(B, 3, H, W)`, float32 — RGB image batch.
- `H` and `W` must be divisible by 4.
- Pixel values normalized to roughly zero mean and unit std.

## Output Format
`stem_out`: `(B, 48, H/4, W/4)`, float32 — low-level features after stem downsampling.
`lcs_out`: `(B, 96, H/8, W/8)`, float32 — after first LCS block with stride 2.

## Constraints
- Stem: Conv(3→24, 3×3, stride=2, padding=1) → BN → ReLU → Conv(24→48, 3×3, stride=2, padding=1) → BN → ReLU.
- LCS Block: for S-size, expand to (96, dw_ks=3). Structure: Conv(48→96, 1×1) → BN → ReLU → DWConv(96, 3×3, stride=2, g=96) → BN → ReLU → Conv(96→96, 1×1) → BN → ReLU.
- Output shape must match exactly `(B, 96, H/8, W/8)`.

## Example
```python
stem = HGNetV2Stem()
x = torch.randn(2, 3, 256, 256)
s_out, l_out = stem(x)
assert s_out.shape == (2, 48, 64, 64)
assert l_out.shape == (2, 96, 32, 32)
```

## Hints
- Use `groups=in_channels` for depthwise convolution.
- `Conv-BN-ReLU` is a recurring 3-layer pattern — make a helper function.
- Choose `bias=False` when followed by BatchNorm (BN absorbs bias).

## Checker
```bash
python checker.py
```
