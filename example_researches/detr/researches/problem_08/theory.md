# Problem 08 Theory - Bipartite Cost Matrix

## Core Definitions
- Matching cost quantifies how suitable prediction $i$ is for target $j$.
- Hungarian solver uses this matrix for globally optimal assignment.

## Variables and Shape Dictionary
- $N$: number of predictions/queries.
- $M$: number of targets.
- $K$: number of foreground classes.
- $\hat{p}_i(c_j)$: predicted probability that query $i$ has target class $c_j$.
- $\hat{b}_i, b_j$: predicted and target boxes.
- Cost matrix $C \in \mathbb{R}^{N\times M}$.

## Main Equations (LaTeX)
$$
C_{cls}(i,j) = -\hat{p}_i(c_j)
$$
$$
C_{L1}(i,j) = \|\hat{b}_i - b_j\|_1
$$
$$
C(i,j)=\lambda_{cls}C_{cls}(i,j)+\lambda_{L1}C_{L1}(i,j)+\lambda_{giou}C_{giou}(i,j)
$$

## Step-by-Step Derivation or Computation Flow
1. Softmax logits to class probabilities.
2. Gather class probability per target label for all $(i,j)$ pairs.
3. Compute pairwise L1 distances between boxes.
4. Compute pairwise geometric penalty via GIoU loss term.
5. Weighted sum all terms into a single matrix.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Inputs: logits $(N,K+1)$, boxes $(N,4)$, target labels $(M,)$, target boxes $(M,4)$.
- Class cost: $(N,M)$.
- L1 cost: $(N,M)$.
- GIoU cost: $(N,M)$.
- Final cost matrix: $(N,M)$.

Worked mini-example:
- If $N=100, M=12$, all cost components and final matrix are $(100,12)$.

## Practical Interpretation
- Classification term says what object it is.
- L1/GIoU terms say where and how well geometry matches.
- Good matching requires both semantic and geometric agreement.