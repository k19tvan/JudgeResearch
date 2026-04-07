# Problem 10 Questions

## Multiple Choice

1. Why is `bias=False` used in convolution layers followed by BatchNorm?
   A. Bias and BatchNorm's beta parameter are redundant — BN absorbs any constant bias
   B. It's required by GPU optimization
   C. Bias causes NaN during training
   D. Removing bias reduces overfitting

2. In a Depthwise Convolution with 96 channels and a 3×3 kernel, how many parameters does it have?
   A. `96 × 96 × 3 × 3 = 82,944`
   B. `96 × 3 × 3 = 864`
   C. `96 × 1 × 3 × 3 = 864`
   D. `96 × 3 = 288`

3. What is the output spatial size after HGNetV2Stem processes `(B, 3, 640, 640)`?
   A. `(B, 96, 160, 160)` for LCS out
   B. `(B, 96, 80, 80)` for LCS out
   C. `(B, 48, 80, 80)` for LCS out
   D. `(B, 96, 320, 320)` for LCS out

4. Why does the LCS block use a 1×1 pointwise conv BEFORE the depthwise conv?
   A. To resize spatial dimensions
   B. To mix channels before spatial filtering (allowing cross-channel interactions that DW cannot do)
   C. To add bias terms
   D. To apply BatchNorm efficiently

5. The overall downsampling factor of Stem + LCS Block from input `(H, W)` is:
   A. 2×
   B. 4×
   C. 8×
   D. 16×

## Answer Key
1.A 2.C 3.B 4.B 5.C
