# Problem 09 Theory - Transformer Decoder Layer with Target Gating

## Core Definitions
- **Decoder Layer**: Three sequential sub-layers: Self-Attention (refines object queries among themselves), Cross-Attention (fetches relevant image context for each query), FFN (mixes extracted information).
- **Target Gating Layer (TGL)**: D-FINE's novel contribution at the decoder level. Instead of a simple residual `tgt = tgt + cross_out`, TGL applies learned sigmoid gates that allow queries to adaptively **blend** their prior state with newly cross-attended information. This prevents information entanglement when queries are reassigned between objects across layers.
- **Key insight from paper**: The TGL prevents a query from being "confused" when it was tracking one target in layer `l-1` and needs to switch to a different target in layer `l`. The gate values allow it to selectively erase prior information.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `tgt` | `(B, N, d)` | Object queries |
| `memory` | `(B, T, d)` | Encoder spatial features |
| `q_sa, k_sa` | `(B, N, d)` | Self-attn q=k = tgt + qpos |
| `cross_out` | `(B, N, d)` | Output of cross-attention |
| `gate_in` | `(B, N, 2d)` | Concatenation [tgt, cross_out] |
| `gate1, gate2` | `(B, N, d)` | Sigmoid gates in [0,1] |
| `tgt_gated` | `(B, N, d)` | After gating: gate1⊙tgt + gate2⊙cross_out |

## Main Equations (LaTeX)

**Self-Attention sub-layer (Pre-LN):**
$$ \text{tgt} \mathrel{+}= \text{Dropout}(\text{MHA}(q_s, k_s, \text{norm1(tgt)})) $$

**Cross-Attention:**
$$ \text{cross\_out} = \text{MHA}(\text{tgt}+\text{qpos},\ \text{memory}+\text{mpos},\ \text{memory}) $$

**Target Gating:**
$$ g_1 = \sigma\!\left([\text{tgt},\ \text{cross\_out}]\, W_1^T + b_1\right),\quad g_2 = \sigma\!\left([\text{tgt},\ \text{cross\_out}]\, W_2^T + b_2\right) $$
$$ \text{tgt} = g_1 \odot \text{tgt} + g_2 \odot \text{cross\_out} $$

**FFN sub-layer (Post-gate):**
$$ \text{tgt} \mathrel{+}= \text{Dropout}(\text{FFN}(\text{LayerNorm}(\text{tgt}))) $$

## Step-by-Step Computation Flow
1. (Pre-LN Self-Attn) `n1=LN(tgt)`. `q=k=n1+qpos`, `v=n1`. `sa_out=MHA(q,k,v)`. `tgt=tgt+drop(sa_out)`.
2. (Cross-Attn) `n2=LN(tgt)`. `q=n2+qpos`, `k=memory+mpos`, `v=memory`. `cross_out, _ = MHA(q,k,v)`.
3. (Target Gating) `cat=torch.cat([tgt, cross_out], -1)`. `g1=sigmoid(Lin1(cat))`. `g2=sigmoid(Lin2(cat))`. `tgt = g1*tgt + g2*cross_out`.
4. (Pre-LN FFN) `n3=LN(tgt)`. `ffn_out=Lin_2(drop(GELU(Lin_1(n3))))`. `tgt=tgt+drop(ffn_out)`.
5. Return `tgt`.

## Tensor Shape Flow
```
tgt: (B, N, d)     memory: (B, T, d)
  ↓ Self-Attn (Pre-LN)
tgt: (B, N, d)
  ↓ Cross-Attn
cross_out: (B, N, d)
  ↓ cat([tgt, cross_out], -1)
gate_in: (B, N, 2d) → gates → g1, g2: (B, N, d)
  ↓ tgt = g1*tgt + g2*cross_out
tgt: (B, N, d)
  ↓ FFN (Pre-LN) + residual
out: (B, N, d)
```

## Practical Interpretation
The Target Gating Layer is the paper's ablation Table 3 step "+TargetGatingLayers" which recovers AP from 52.4% to 52.8% after removing decoder projection layers. The gate values reflect how much each query "needs to change" — queries already well-localized will primarily gate through their prior state (high g1), while confused queries heavily gate the new cross-attended context (high g2).
