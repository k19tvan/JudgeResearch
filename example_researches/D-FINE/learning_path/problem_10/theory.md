# Problem 10 Theory - HGNetV2-S Stem and Depthwise Conv Block

## Core Definitions
- **HGNetV2**: A CNN backbone purpose-built for fast GPU inference. Replaces expensive 3×3 group convolutions with lightweight DWConv (depthwise) + pointwise 1×1 convolutions in an "inverted bottleneck" pattern.
- **Stem**: The initial downsampling stage that converts raw RGB `(H,W)` to a compact feature map `(H/4, W/4)` with 48 channels, using stacked 3×3 strided convolutions.
- **LCS Block (Local Channel Sharing)**: The core HGNetV2 block. Pointwise → Depthwise → Pointwise structure with optional stride for spatial downsampling.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `x` | `(B, 3, H, W)` | RGB input |
| After stem conv1 | `(B, 24, H/2, W/2)` | First stride-2 conv |
| After stem conv2 | `(B, 48, H/4, W/4)` | Second stride-2 conv |
| After LCS pw1 | `(B, 96, H/4, W/4)` | 1×1 channel expansion |
| After LCS dw | `(B, 96, H/8, W/8)` | DW 3×3 stride-2 downsampling |
| After LCS pw2 | `(B, 96, H/8, W/8)` | 1×1 channel refinement |

## Main Equations (LaTeX)

**Depthwise Convolution (channel-wise spatial filter):**
$$ \text{DWConv}(X)_c = X_c \star K_c, \quad K_c \in \mathbb{R}^{k \times k} $$

**Inverted Bottleneck FLOPs reduction:**
Standard 3×3 conv (C→C): $O(H \cdot W \cdot C^2 \cdot k^2)$
DW + PW (C→C): $O(H \cdot W \cdot C \cdot k^2) + O(H \cdot W \cdot C^2)$
Ratio: $\approx \frac{1}{C} + \frac{1}{k^2}$ ← dramatically cheaper for large C.

**Conv-BN-ReLU block:**
$$ \text{CBR}(X) = \text{ReLU}(\text{BN}(\text{Conv}(X))) $$

## Step-by-Step Computation Flow
**Stem:**
1. `x → Conv(3, 24, k=3, s=2, p=1) → BN → ReLU` → `(B, 24, H/2, W/2)`.
2. `→ Conv(24, 48, k=3, s=2, p=1) → BN → ReLU` → `(B, 48, H/4, W/4)`. [stem_out]

**LCS Block:**
3. `stem_out → Conv(48, 96, k=1) → BN → ReLU` → `(B, 96, H/4, W/4)` [PW expand].
4. `→ DWConv(96, k=3, s=2, p=1, g=96) → BN → ReLU` → `(B, 96, H/8, W/8)` [DW stride-2].
5. `→ Conv(96, 96, k=1) → BN → ReLU` → `(B, 96, H/8, W/8)`. [lcs_out]

## Tensor Shape Flow
```
Input: (B, 3, H, W)
  ↓ CBR(3→24, stride=2)
(B, 24, H/2, W/2)
  ↓ CBR(24→48, stride=2)
stem_out: (B, 48, H/4, W/4)
  ↓ PW CBR(48→96)
(B, 96, H/4, W/4)
  ↓ DW CBR(96, stride=2)
(B, 96, H/8, W/8)
  ↓ PW CBR(96→96)
lcs_out: (B, 96, H/8, W/8)
```

## Practical Interpretation
The Stem's dual-stride design (2×downsampling by 2) is a design choice from ResNet — using two 3×3 convolutions is receptively equivalent to a single 7×7 conv while being more parameter-efficient and faster in practice. The LCS block's depthwise approach reduces compute by ~8× compared to regular 3×3 convolutions while maintaining spatial feature quality through the flanking pointwise convolutions.
