# Problem 03 Theory - Focal Loss

## Core Definitions
Focal loss is a modification of standard Cross-Entropy Loss that down-weights the loss assigned to well-classified examples (the "easy" negatives) and focuses training on a sparse set of hard examples.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| `inputs` | `(N, C)` | Unnormalized logit predictions. |
| `targets` | `(N, C)` | One-hot binary (0 or 1) classification targets. |
| `prob` | `(N, C)` | Sigmoid activated probabilities $[0, 1]$. |
| `p_t` | `(N, C)` | The probability of the true class. |
| `ce_loss` | `(N, C)` | Standard binary cross-entropy loss. |

## Main Equations (LaTeX)

Standard Cross Entropy is given by:
$$ CE(p_t) = -\log(p_t) $$
Where $p_t$ simplifies to $p$ if $y=1$, and $1-p$ if $y=0$.

Focal Loss introduces a modulating factor $(1 - p_t)^\gamma$:
$$ FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$

When an example is misclassified and $p_t$ is small, the modulating factor is near $1$ and the loss is unaffected. As $p_t \to 1$, the factor goes to $0$ and the loss for well-classified examples is down-weighted.

## Step-by-Step Derivation or Computation Flow
1. Use `sigmoid()` to get raw probabilities $p$.
2. Calculate the base `BCEWithLogitsLoss`. This is numerically stable.
3. Compute $p_t$, the probability of the true label using $p \cdot y + (1-p) \cdot (1-y)$.
4. Compute the class weight $\alpha_t$ using $\alpha \cdot y + (1-\alpha) \cdot (1-y)$.
5. Calculate the final focal penalty: $\alpha_t \cdot (1 - p_t)^\gamma \cdot \text{bce}$.
6. Return the raw unreduced tensor.

## Practical Interpretation
In an image containing $10,000$ anchor boxes but only $5$ actual objects, $9,995$ boxes represent the background (negative). Standard loss would let the sheer volume of these easy negatives overshadow the $5$ real foreground signals. Focal loss silences those $9,995$ boxes aggressively.