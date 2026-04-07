# Problem 06 Theory - 2D Sine Positional Encoding

## Core Definitions
- **Positional Encoding (PE)**: A fixed (non-learned) addition to input embeddings that encodes spatial location. Without PE, transformers cannot distinguish between different spatial positions.
- **Sinusoidal PE**: Uses sine/cosine at varying frequencies to encode positions. Low-frequency sinusoids capture coarse global position; high-frequency ones encode fine local detail.
- **2D extension**: Images have both x and y spatial dimensions. The `d_model` embedding channels are split: half encode y-position, half encode x-position.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `x` | `(B, C, H, W)` | Input feature map |
| `y_embed` | `(B, H, W)` | Normalized y-position grid `[0,1]` |
| `x_embed` | `(B, H, W)` | Normalized x-position grid `[0,1]` |
| `dim_t` | `(d_model//2,)` | Frequency index `[0, 1, ..., d_model//2-1]` |
| `W(i)` | scalar per i | Temperature power: `temperature^(2i/d_model)` |
| `pos_x` | `(B, H, W, d_model//2)` | Sine/Cosine encoded x positions |
| `pos_y` | `(B, H, W, d_model//2)` | Sine/Cosine encoded y positions |
| `pos` | `(B, H*W, d_model)` | Final positional encodings |

## Main Equations (LaTeX)

**1D Sinusoidal PE at position $p$ and dimension $2i$ (even):**
$$ \text{PE}(p, 2i) = \sin\!\left(\frac{p}{\text{temperature}^{2i/d}}\right) $$

**1D Sinusoidal PE at dimension $2i+1$ (odd):**
$$ \text{PE}(p, 2i+1) = \cos\!\left(\frac{p}{\text{temperature}^{2i/d}}\right) $$

**Normalized grid position:**
$$ p_x = \frac{x\text{-index} + 0.5}{W}, \quad p_y = \frac{y\text{-index} + 0.5}{H} $$

**2D concatenation:**
$$ \text{PE}_{2D}(p_y, p_x) = [\text{PE}(p_x, 0{:}d/2)\ \|\ \text{PE}(p_y, 0{:}d/2)] \in \mathbb{R}^d $$

## Step-by-Step Derivation or Computation Flow
1. Build `y_embed`: cumsum over dim=1 (rows), normalize to [0,1] by dividing by `H`.
2. Build `x_embed`: cumsum over dim=2 (cols), normalize to [0,1] by dividing by `W`.
3. `dim_t = torch.arange(d_model//2)`.
4. `omega = temperature^(2i/d_model)` → shape `(d_model//2,)`.
5. Divide: `x_embed[...,None] / omega[None,None,None,:]` → `(B, H, W, d_model//4)` — then interleave sin/cos.
6. Actually simpler: compute `arg_x = x_embed[...,None] / omega` → `(B,H,W, d_model//4)`, then `sin(arg_x), cos(arg_x)` → stack → reshape to `(B,H,W, d_model//2)`.
7. Same for y. Then cat(pos_x, pos_y, dim=-1) → `(B,H,W,d_model)`.
8. Flatten: `.flatten(1,2)` → `(B, H*W, d_model)`.

## Tensor Shape Flow
```
x: (B, C, H, W)
   ↓ build grids
x_embed, y_embed: (B, H, W)
   ↓ broadcast with dim_t (d_model//4,)
arg_x, arg_y: (B, H, W, d_model//4)
   ↓ sin/cos + stack
pos_x, pos_y: (B, H, W, d_model//2)
   ↓ cat
(B, H, W, d_model) → flatten(1,2) → (B, H*W, d_model)
```

## Practical Interpretation
The lower (even) dimension indices in `dim_t` correspond to large-scale sinusoids (slow variation across the image) while higher indices create fine-grained position discrimination. Together, every spatial patch in the `H×W` grid gets a unique `d_model`-dimensional fingerprint that the transformer self-attention can use to spatially relate image regions. In D-FINE, this PE is added to the flattened backbone feature map before the Hybrid Encoder's transformer layers.

**Mini-example (d_model=4, temperature=10000):**
- d_model//4 = 1 dimension per sin/cos per axis.
- Position (x=0.3, y=0.7): `arg_x = 0.3/1 = 0.3`. `pos_x = [sin(0.3), cos(0.3)]`.
- `arg_y = 0.7`. `pos_y = [sin(0.7), cos(0.7)]`.
- Final: `[sin(0.3), cos(0.3), sin(0.7), cos(0.7)]` ∈ ℝ⁴.
