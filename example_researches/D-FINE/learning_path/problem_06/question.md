# Problem 06 Questions

## Multiple Choice

1. Why do DETR-family models need explicit positional encodings at the transformer input?
   A. Transformers apply convolutions and thus have spatial inductive bias
   B. Self-attention is permutation-invariant — it cannot distinguish token positions without PE
   C. Positional encodings prevent gradient vanishing
   D. They replace the need for batch normalization

2. In `PositionEmbeddingSine2D`, the `d_model` dimension is split so that:
   A. Even indices encode x, odd indices encode y
   B. First half encodes x-position, second half encodes y-position
   C. First quarter is x-sin, second quarter is x-cos, third is y-sin, fourth is y-cos
   D. The split is arbitrary and configurable

3. Why does the sinusoidal formula use the `temperature` (default 10000)?
   A. To normalize values to [-1, 1]
   B. To control the range of frequencies — lower temperature → higher overall frequency
   C. To control the signal-to-noise ratio during training
   D. To set the learning rate schedule

4. What does `dim_t = torch.arange(d_model // 4)` represent in the implementation?
   A. The number of attention heads
   B. The frequency index for each sin/cos pair before interleaving
   C. The spatial grid indices
   D. The channel indices of the backbone output

5. If `d_model=8`, how many sin/cos pairs are used to encode the x-position?
   A. 8
   B. 4
   C. 2
   D. 1

## Answer Key
1.B 2.B 3.B 4.B 5.C
