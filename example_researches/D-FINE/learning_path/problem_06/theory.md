# Problem 06 Theory - Contrastive Denoising (CDN) Concept

## Core Definitions
Adding noise to true boxes to accelerate DETR convergence via auxiliary tasks.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| `input` | `(B, C, H, W)` | Standard feature map block. |

## Main Equations (LaTeX)
$$ Y = \text{Layer}(X) $$
*(Specific equations will be populated during the detailed problem phase).*

## Step-by-Step Derivation or Computation Flow
1. Load features.
2. Apply transformation.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- `Input`: Tensor `(B, *dims)`
- `Output`: Transformed Tensor `(B, *new_dims)`

## Practical Interpretation
This module handles a crucial step in the end-to-end target sequence.
