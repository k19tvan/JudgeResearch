# Problem 05 Theory - Transformer Encoder Layer

## Core Definitions
- Encoder layer = self-attention block + feed-forward block, each wrapped by residual connections and normalization.
- In DETR, encoder processes flattened image tokens from backbone features.

## Variables and Shape Dictionary
- $N$: token count ($H\times W$ after flattening).
- $B$: batch size.
- $C$: hidden dimension.
- $P$: positional encoding $(N,B,C)$.
- $X$: input sequence $(N,B,C)$.
- FFN expansion dimension: $C_{ff}$ (e.g., 2048).

## Main Equations (LaTeX)
$$
X_1 = X + \text{Dropout}(\text{MHA}(Q=X+P, K=X+P, V=X))
$$
$$
X_2 = \text{Norm}(X_1)
$$
$$
Y = X_2 + \text{Dropout}(W_2\,\phi(W_1 X_2 + b_1)+b_2)
$$

## Step-by-Step Derivation or Computation Flow
1. Add positional encoding to queries and keys.
2. Compute self-attention over all spatial tokens.
3. Add residual from original input and normalize.
4. Apply two-layer FFN with nonlinearity.
5. Add second residual and normalize.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input: $(N,B,C)$.
- Attention output: $(N,B,C)$.
- Post-attention normalized: $(N,B,C)$.
- FFN hidden: $(N,B,C_{ff})$.
- Final output: $(N,B,C)$.

Worked mini-example:
- If $H=W=20$, then $N=400$.
- With $B=2, C=256, C_{ff}=2048$, FFN hidden tensor is $(400,2,2048)$.

## Practical Interpretation
- Self-attention injects global context: each token can use information from all other tokens.
- FFN mixes channels per token to improve representation power.
- Residual design stabilizes deep stacking of encoder layers.