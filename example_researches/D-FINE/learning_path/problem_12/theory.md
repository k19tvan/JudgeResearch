# Problem 12 Theory - Training and Evaluation Loop

## Core Definitions
- **Training Loop**: One epoch iterates over all batches. For each batch: forward pass → loss computation → backpropagation → parameter update.
- **AdamW Optimizer**: Adam with decoupled weight decay. Combines adaptive per-parameter learning rates (Adam) with L2 regularization applied directly to parameters (not via gradient, hence "decoupled").
- **Evaluation Loop**: Runs the model in `eval()` mode with `torch.no_grad()` — disabling dropout and using running statistics for BatchNorm — to measure detection quality without training overhead.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `images` | `(B, 3, 256, 256)` | Batch of input images |
| `targets` | list of B dicts | GT labels and boxes per image |
| `out["pred_logits"]` | `(B, N, C)` | Raw classification scores |
| `out["pred_boxes"]` | `(B, N, 4)` | Predicted boxes cxcywh [0,1] |
| `indices` | list of B tuples | Matcher assignment |
| `losses` | dict of scalars | `loss_vfl`, `loss_bbox`, `loss_giou` |
| `total_loss` | scalar | Sum of all losses |

## Main Equations (LaTeX)

**Total loss (weighted sum from SetCriterion):**
$$ \mathcal{L} = \lambda_{vfl} \mathcal{L}_{vfl} + \lambda_{bbox} \mathcal{L}_{bbox} + \lambda_{giou} \mathcal{L}_{giou} $$

**AdamW update rule:**
$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $$
$$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$
$$ \theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \alpha \lambda \theta_{t-1} $$

**Mean IoU metric (simplified):**
$$ \text{mIoU} = \frac{1}{B} \sum_{b} \frac{1}{T_b} \sum_{i=1}^{T_b} \max_{j} \text{IoU}(\hat{b}_{bj}, b^{GT}_{bi}) $$

## Step-by-Step Training Loop
1. `model.train()`.
2. For each `(images, targets)` batch:
   a. `optimizer.zero_grad()`.
   b. `images, targets → device`.
   c. `out = model(images)`.
   d. `indices = matcher(out, targets)["indices"]`.
   e. `losses = criterion(out, targets, indices)`.
   f. `total_loss = sum(losses.values())`.
   g. `total_loss.backward()`.
   h. `optimizer.step()`.
3. Return averaged loss metrics.

## Step-by-Step Eval Loop
1. `model.eval()`.
2. `with torch.no_grad():` for each batch:
   a. `out = model(images)`.
   b. Get top predictions by class confidence.
   c. Compute IoU between each GT box and all predictions, take max.
3. Average max IoU across all GT instances.

## Practical Interpretation
The training loop's elegance comes from the fact that loss computation and matching are fully differentiable through the SetCriterion (after index selection). Each iteration, the Hungarian Matching decides WHICH predictions supervise WHICH GTs — and this decision changes dynamically as the model improves. Early in training, predictions are nearly random, so matching is unstable; after several epochs, predictions stabilize, and matching converges to consistent assignments that drive precise box refinement.
