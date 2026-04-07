# Problem 07 Theory - Multi-Head Self-Attention

## Core Definitions
- **Self-Attention**: Each position in the sequence queries all positions (including itself) to compute a weighted mixture of value vectors. This allows global receptive field in a single layer.
- **Multi-Head**: Running H independent attention operations in parallel, each with `d_k = d_model/H` dimensions. Heads can specialize — some may attend to local edges, others to global semantic relations.
- **Scaled Dot-Product**: Dividing by `sqrt(d_k)` prevents dot products from growing large with dimensionality (which pushes softmax into saturation regions with near-zero gradients).

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `q,k,v` | `(B, T, d_model)` | Input sequences |
| `Q,K,V` | `(B, H, T, d_k)` | Projected and head-split |
| `d_k` | scalar | Per-head key dimension = `d_model // H` |
| `scores` | `(B, H, T_q, T_k)` | Raw dot-product similarities |
| `attn` | `(B, H, T_q, T_k)` | Softmax attention weights |
| `ctx` | `(B, H, T_q, d_k)` | Attended value context |
| `out` | `(B, T_q, d_model)` | Final projected output |

## Main Equations (LaTeX)

**Scaled Dot-Product Attention:**
$$ \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

**Multi-Head:**
$$ \text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V) $$

$$ \text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O $$

**Score scaling (motivation):**
If entries of $Q, K$ are i.i.d. with mean 0, variance 1, then $Q \cdot K$ has variance $d_k$.
Dividing by $\sqrt{d_k}$ restores unit variance → stable softmax gradients.

## Step-by-Step Derivation or Computation Flow
1. Project: `Q = W_q(q)` → `(B, T_q, d_model)`. Same for K, V.
2. Split heads: reshape to `(B, T, H, d_k)` then `.transpose(1,2)` → `(B, H, T, d_k)`.
3. Score: `scores = Q @ K.transpose(-2,-1) / sqrt(d_k)` → `(B, H, T_q, T_k)`.
4. Mask: if `key_padding_mask` given → `scores.masked_fill(mask[...,None,None,:], -inf)`.
5. Softmax: `attn = scores.softmax(dim=-1)`. Dropout on attn.
6. Context: `ctx = attn @ V` → `(B, H, T_q, d_k)`.
7. Merge: `.transpose(1,2).contiguous().reshape(B, T_q, d_model)`.
8. Output: `out = W_o(ctx_merged)`.

## Tensor Shape Flow
```
q: (B,T_q,d) → W_q → Q: (B,T_q,d) → reshape → (B,H,T_q,d_k)
k: (B,T_k,d) → W_k → K: (B,T_k,d) → reshape → (B,H,T_k,d_k)
v: (B,T_k,d) → W_v → V: (B,T_k,d) → reshape → (B,H,T_k,d_k)

Q @ Kᵀ / sqrt(d_k)  →  scores: (B, H, T_q, T_k)
     ↓ softmax
attn: (B, H, T_q, T_k)
     ↓ @ V
ctx: (B, H, T_q, d_k) → merge → (B, T_q, d) → W_o → out: (B, T_q, d)
```

## Practical Interpretation
In D-FINE's Transformer Encoder, self-attention is applied over the flattened feature map `(B, H*W, d_model)`. Each pixel patch can directly attend to any other patch in one layer — this is fundamental to DETR's ability to suppress duplicate predictions by different queries. In the Decoder, cross-attention is used: queries are object slots `(B, N, d_model)`, keys/values are image features `(B, H*W, d_model)`. Each object query learns WHERE in the image to focus.

**Mini-example (H=2, d_model=4, d_k=2):**
- Q = [[1,0,1,0]], K = [[1,0,0,1], [0,1,1,0]] → after projection (d_k=2), scores = Q_h @ K_h^T / sqrt(2) → 2 scores → softmax → weighted V → concat → output.
