# Problem 10 Theory - Set Criterion Loss

## Core Definitions
- Set criterion computes supervised loss after one-to-one matching.
- Classification is over all queries; box losses are only for matched pairs.

## Variables and Shape Dictionary
- $N_q$: number of queries.
- $N_{gt}$: number of matched targets.
- $\hat{p}_i$: class distribution for query $i$.
- $y_i$: target class for query $i$ (foreground or no-object).
- $\hat{b}_i, b_j$: predicted and target boxes.
- Logits shape: $(B,N_q,K+1)$.
- Boxes shape: $(B,N_q,4)$.

## Main Equations (LaTeX)
$$
\mathcal{L}_{cls}= -\sum_{i=1}^{N_q} w_{y_i}\log\hat{p}_i(y_i)
$$
$$
\mathcal{L}_{L1}=\frac{1}{N_{gt}}\sum_{(i,j)\in\sigma}\|\hat{b}_i-b_j\|_1
$$
$$
\mathcal{L}_{total}=\lambda_{cls}\mathcal{L}_{cls}+\lambda_{L1}\mathcal{L}_{L1}+\lambda_{giou}\mathcal{L}_{giou}
$$

## Step-by-Step Derivation or Computation Flow
1. Build target class tensor for all queries (default no-object).
2. Overwrite matched query positions with matched GT labels.
3. Compute cross-entropy classification loss over all queries.
4. Gather matched boxes and compute L1 + GIoU terms.
5. Combine weighted losses into dictionary and scalar training objective.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input logits: $(B,N_q,K+1)$.
- Target classes: $(B,N_q)$.
- Matched boxes: $(N_{gt},4)$ vs $(N_{gt},4)$.
- Output losses: scalar tensors.

Worked mini-example:
- If $B=2$, $N_q=100$, matched count across batch $N_{gt}=14$.
- Classification supervises 200 slots; box losses supervise 14 matched slots.

## Practical Interpretation
- This objective balances foreground learning and background suppression.
- Matching-aware supervision is what enables NMS-free set prediction.