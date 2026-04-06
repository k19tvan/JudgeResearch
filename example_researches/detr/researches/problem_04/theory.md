# Problem 04 Theory - Attention Mechanics

## Core Definitions
- Scaled dot-product attention computes weighted sums of values based on query-key similarity.
- Multi-head attention runs this process in parallel subspaces and concatenates outputs.

## Variables and Shape Dictionary
- $N$: sequence length.
- $B$: batch size.
- $C$: model dimension.
- $h$: number of heads.
- $d_h=C/h$: per-head dimension.
- $Q,K,V$: tensors of shape $(N,B,C)$ before head split.
- $P_Q, P_K$: positional encodings for Q and K with shape $(N,B,C)$.
- Output shape: $(N,B,C)$.

## Main Equations (LaTeX)
$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$
$$
\tilde{Q}=Q+P_Q, \quad \tilde{K}=K+P_K
$$
$$
\text{MHA}(Q,K,V)=\text{Concat}(head_1,\dots,head_h)W_O
$$

## Step-by-Step Derivation or Computation Flow
1. Add positional tensors only to Q and K.
2. Project Q/K/V to per-head spaces.
3. Compute scaled similarity matrix per head.
4. Apply softmax along key dimension to obtain attention weights.
5. Weighted-sum values, concatenate heads, apply output projection.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input Q/K/V: $(N,B,C)$.
- After split: $(B,h,N,d_h)$.
- Attention logits: $(B,h,N,N)$.
- Attention output per head: $(B,h,N,d_h)$.
- Final output: $(N,B,C)$.

Worked mini-example:
- $N=400, B=2, C=256, h=8 \Rightarrow d_h=32$.
- Per head logits shape is $(2,8,400,400)$.

## Practical Interpretation
- Position on Q/K changes where attention looks, while unmodified V preserves feature content.
- This design is central to DETR: geometry steers attention, semantics flow through values.