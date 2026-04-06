# Problem 03 Theory - 2D Sine Positional Encoding

## Core Definitions
- Transformers are permutation-invariant; positional encoding injects order/geometry.
- For images, position is 2D: row ($y$) and column ($x$).

## Variables and Shape Dictionary
- $B$: batch size.
- $C$: embedding dimension, even.
- $H, W$: spatial height and width.
- $p$: normalized coordinate value.
- $\tau$: temperature constant (usually 10000).
- Mask shape: $(B, H, W)$.
- Positional output shape: $(B, C, H, W)$.

## Main Equations (LaTeX)
$$
PE_{p,2i}=\sin\left(\frac{p}{\tau^{2i/d}}\right), \quad PE_{p,2i+1}=\cos\left(\frac{p}{\tau^{2i/d}}\right)
$$
$$
\mathbf{e}_x \in \mathbb{R}^{C/2}, \quad \mathbf{e}_y \in \mathbb{R}^{C/2}, \quad \mathbf{e}_{2D}=[\mathbf{e}_y;\mathbf{e}_x] \in \mathbb{R}^{C}
$$
$$
P \in \mathbb{R}^{B\times C\times H\times W}
$$

## Step-by-Step Derivation or Computation Flow
1. Build valid-coordinate accumulators from mask (ignore padded positions).
2. Optionally normalize coordinates to $[0, 2\pi]$.
3. Compute frequency vector for channels.
4. Apply sine to even channel indices and cosine to odd indices.
5. Concatenate y-encoding and x-encoding along channel axis.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input mask: $(B,H,W)$.
- Intermediate x/y coordinates: $(B,H,W)$ each.
- Intermediate expanded frequencies: $(B,H,W,C/2)$.
- Output positional tensor: $(B,C,H,W)$.

Worked mini-example:
- Let $B=2, C=256, H=20, W=30$.
- x-embedding shape $(2,20,30,128)$, y-embedding shape $(2,20,30,128)$.
- Concatenated output becomes $(2,256,20,30)$.

## Practical Interpretation
- Low-frequency channels capture coarse location; high-frequency channels capture fine location.
- Adding this encoding to Q/K lets attention distinguish where features come from in the image.