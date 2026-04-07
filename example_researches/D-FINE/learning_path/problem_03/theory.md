# Problem 03 Theory - Hungarian Matcher (Bipartite Matching)

## Core Definitions
- **Set-based detection**: DETR removes anchors and NMS by treating detection as a set prediction problem. Predictions are a SET of N object candidates; matching is required to assign each unique ground truth to exactly one prediction.
- **Linear Sum Assignment (Hungarian Algorithm)**: Given an N×T cost matrix, finds the 1-to-1 assignment of N workers to T tasks minimizing total cost. In D-FINE this operates on cost_class + cost_L1 + cost_giou.
- **Focal cost**: Reweights classification confidence by `(1-p)^gamma` to emphasize hard examples. This prevents the matcher from always choosing predictions with the highest class confidence.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `pred_logits` | `(B, N, C)` | Raw class logits for N queries |
| `out_prob` | `(B*N, C)` | Sigmoid probabilities, flattened |
| `out_bbox` | `(B*N, 4)` | Predicted boxes, flattened |
| `tgt_ids` | `(T_total,)` | All target class ids concatenated |
| `tgt_bbox` | `(T_total, 4)` | All target boxes concatenated |
| `cost_class` | `(B*N, T_total)` | Focal classification cost |
| `cost_bbox` | `(B*N, T_total)` | L1 box cost |
| `cost_giou` | `(B*N, T_total)` | GIoU cost (negated) |
| `C` | `(B, N, T_total)` | Final combined cost matrix |
| `indices` | list of B `(src, tgt)` tuples | Optimal assignment per image |

## Main Equations (LaTeX)

**Sigmoid Focal Classification Cost:**
$$ p_{ij} = \sigma(\text{logits}_{ij}) $$
$$ c^{cls}_{ij} = \alpha (1 - p_{ij})^\gamma \cdot (-\log p_{ij}) - (1-\alpha) p_{ij}^\gamma \cdot (-\log(1-p_{ij})) $$

**L1 Box Cost:**
$$ c^{L1}_{ij} = \| \hat{b}_i - b_j \|_1 $$

**GIoU Cost:**
$$ c^{GIoU}_{ij} = -\text{GIoU}(\hat{b}_i, b_j) $$

**Combined Cost:**
$$ C_{ij} = \lambda_1 c^{L1}_{ij} + \lambda_2 c^{cls}_{ij} + \lambda_3 c^{GIoU}_{ij} $$

**Hungarian Assignment:**
$$ \hat{\sigma} = \arg\min_{\sigma \in \mathcal{P}(N)} \sum_{i=1}^{T} C_{i,\sigma(i)} $$

## Step-by-Step Derivation or Computation Flow
1. Sigmoid `pred_logits` → `out_prob` `(B*N, C)`.
2. Concat all `tgt_ids` → `(T_total,)`. Index `out_prob` at `tgt_ids` columns → `(B*N, T_total)`.
3. Compute `pos_cost` and `neg_cost` using focal weighting. Subtract to get `cost_class`.
4. `cost_bbox = cdist(out_bbox, tgt_bbox, p=1)` → `(B*N, T_total)`.
5. Convert both to `xyxy`, compute `cost_giou = -GIoU(...)` → `(B*N, T_total)`.
6. `C = w1*cost_bbox + w2*cost_class + w3*cost_giou`, reshape to `(B, N, T_total)`.
7. Split columns by `sizes=[T_i]`, call `linear_sum_assignment` per image.
8. Convert numpy results to `torch.int64` tensors.

## Tensor Shape Flow (Input → Intermediate → Output)
```
pred_logits: (B, N, C) → sigmoid → (B*N, C)
pred_boxes:  (B, N, 4) → flatten → (B*N, 4)
tgt_ids:     concat → (T_total,)
                 ↓ column-select + focal
cost_class: (B*N, T_total)
cost_bbox:  (B*N, T_total)   ← cdist L1
cost_giou:  (B*N, T_total)   ← -GIoU
                 ↓ weighted sum + reshape
C: (B, N, T_total) → split → linear_sum_assignment per image
output: list of B tuples [(src_idx, tgt_idx), ...]
```

## Practical Interpretation
With 300 queries and 10 ground-truth boxes, only 10 queries "win" the assignment and receive supervision signals. The other 290 are labeled as "no-object." This clean separation makes DETR end-to-end trainable without anchor/NMS engineering. The temperature of the focal cost (gamma=2) prevents degenerate matchings where predictions with accidentally high confidence dominate the assignment.

**Mini-example:**  
B=1, N=2 predictions, T=1 ground truth.  
Cost matrix C shape: (2, 1).  
`linear_sum_assignment` picks the row with lower cost → exactly 1 prediction is matched.
