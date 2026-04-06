# Problem 10 Theory - Decoupled Decoder Heads

## Core Definitions
Separating features for Classification vs. Regression metrics in the final output heads.

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
