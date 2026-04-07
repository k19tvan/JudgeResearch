# Problem 08 Theory - Transformer Encoder Layer

## Core Definitions
- **Encoder Layer**: Self-contained MHA + FFN block that transforms a sequence in-place. Stacking multiple such layers allows progressive feature refinement.
- **Pre-LayerNorm (Pre-LN)**: Applies LN *before* the attention/FFN sub-layer (inside the residual branch), making training more stable vs. the original Post-LN design.
- **Positional Bias Injection**: In D-FINE, `pos` is added only to queries and keys (not values). This injects spatial awareness into the attention routing without distorting the value content.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `src` | `(B, T, d)` | Input sequence tokens |
| `pos` | `(B, T, d)` | Positional encodings from P06 |
| `norm_src` | `(B, T, d)` | Pre-normalized src |
| `q, k` | `(B, T, d)` | Query = key = norm_src + pos |
| `v` | `(B, T, d)` | Value = norm_src (no position) |
| `attn_out` | `(B, T, d)` | MHA output |
| `ffn_out` | `(B, T, d)` | FFN output |

## Main Equations (LaTeX)

**Pre-LN Self-Attention:**
$$ x' = x + \text{Dropout}\!\left(\text{MHA}(x_{norm} + p,\ x_{norm} + p,\ x_{norm})\right) $$

**Pre-LN FFN:**
$$ x'' = x' + \text{Dropout}\!\left(\text{FFN}(\text{LayerNorm}(x'))\right) $$

**FFN inner computation:**
$$ \text{FFN}(z) = W_2 \cdot \text{GELU}(W_1 z + b_1) + b_2 $$

**GELU activation:**
$$ \text{GELU}(x) \approx x \cdot \sigma(1.702 \cdot x) $$

## Step-by-Step Computation Flow
1. `norm_src = LayerNorm1(src)` → `(B, T, d)`.
2. `q = k = norm_src + pos`, `v = norm_src`.
3. `attn_out, _ = MHA(q, k, v)` → `(B, T, d)`.
4. `src = src + Dropout(attn_out)`.
5. `norm_src2 = LayerNorm2(src)`.
6. `ffn_out = Linear2(Dropout(GELU(Linear1(norm_src2))))`.
7. `src = src + Dropout(ffn_out)`.
8. Return `src`.

## Tensor Shape Flow
```
src: (B, T, d)
  ↓ LayerNorm1
norm_src: (B, T, d) + pos → q, k; norm_src → v
  ↓ MHA(q, k, v)
attn_out: (B, T, d)
  ↓ + residual
src: (B, T, d)
  ↓ LayerNorm2
  ↓ Linear(d→d_ffn) → GELU → Linear(d_ffn→d)
ffn_out: (B, T, d)
  ↓ + residual
out: (B, T, d)
```

## Practical Interpretation
In D-FINE's Hybrid Encoder, encoder layers perform intra-scale self-attention on each of the 3 feature map scales (S=3 encoder layers for small, M=6 for medium, L=3 for large scale). The positional encodings ensure each activation "knows" its 2D grid location before cross-scale fusion. The FFN (4× wider than d_model) then performs a non-linear mixing of attention outputs, acting as a per-token transformation.
