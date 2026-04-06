# Problem 11 Theory - Capstone End-to-End Training Step

## Core Definitions
- One training step integrates all modules: model forward, matcher, criterion, backward, clip, optimizer update.
- Only differentiable parts contribute to gradients; matching is treated as index selection.

## Variables and Shape Dictionary
- $\theta$: model parameters.
- $x$: input batch tensor $(B,3,H,W)$.
- $t$: targets (labels/boxes per sample).
- $f_\theta(x)$: model outputs logits and boxes.
- $\sigma$: Hungarian assignment indices.
- $\eta$: learning rate.
- Total loss: scalar tensor shape $(1,)$ or scalar.

## Main Equations (LaTeX)
$$
\hat{y}=f_\theta(x)
$$
$$
\sigma = \text{HungarianMatch}(\hat{y}, t), \quad \mathcal{L}=\text{SetCriterion}(\hat{y}, t, \sigma)
$$
$$
\theta \leftarrow \theta - \eta\,\text{OptimizerStep}(\nabla_\theta \mathcal{L})
$$

## Step-by-Step Derivation or Computation Flow
1. Reset gradients with zero_grad.
2. Forward pass to obtain predictions.
3. Compute assignment between predictions and targets.
4. Compute weighted set loss.
5. Backpropagate gradients.
6. Clip gradient norm for stability.
7. Update parameters with optimizer step.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input images: $(B,3,H,W)$.
- Model outputs: logits $(B,N_q,K+1)$, boxes $(B,N_q,4)$.
- Matcher outputs: index pairs of shape $(r,)$ per sample.
- Criterion output: scalar losses.
- Optimizer output: updated parameter tensors.

Worked mini-example:
- Let $B=2$, $N_q=100$, $K=80$.
- Forward produces logits $(2,100,81)$ and boxes $(2,100,4)$.
- If total matched objects $r=11$, box terms supervise 11 matched predictions.

## Practical Interpretation
- This step is the smallest executable unit of DETR learning.
- Correct ordering is critical: update before backward or missing zero_grad leads to broken optimization dynamics.