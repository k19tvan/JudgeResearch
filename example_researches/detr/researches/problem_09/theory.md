# Problem 09 Theory - Hungarian Matcher Assignment

## Core Definitions
- Assignment is solved as a linear sum assignment problem (LSAP).
- The objective picks one-to-one matches minimizing total cost.

## Variables and Shape Dictionary
- $C \in \mathbb{R}^{N\times M}$: cost matrix from Problem 08.
- $N$: number of predicted queries.
- $M$: number of targets.
- $\sigma$: assignment mapping from target index to query index.
- Output indices: query index vector and target index vector with equal length $r=\min(N,M)$.

## Main Equations (LaTeX)
$$
\hat{\sigma}=\arg\min_{\sigma}\sum_{j=1}^{M} C_{\sigma(j),j}
$$
$$
\mathcal{P}_{\sigma(i),i}=1 \;\text{iff prediction}\;\sigma(i)\;\text{is assigned to target}\;i
$$
$$
\min_{\mathcal{P}}\sum_{i=1}^{N}\sum_{j=1}^{M} C_{ij}\mathcal{P}_{ij}
\quad\text{s.t. row/column one-to-one constraints}
$$

## Step-by-Step Derivation or Computation Flow
1. Move cost matrix to CPU numpy (for scipy solver).
2. Call linear_sum_assignment.
3. Receive row indices (predictions) and column indices (targets).
4. Convert back to torch tensors on expected device.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input cost matrix: $(N,M)$.
- Solver outputs index vectors: $(r,)$ and $(r,)$.
- Final matched pairs count: $r$.

Worked mini-example:
- If $N=100, M=7$, then $r=7$ matches are returned.

## Practical Interpretation
- This avoids heuristic matching and guarantees globally best supervision pairs.
- Unmatched predictions are treated as no-object in classification supervision.