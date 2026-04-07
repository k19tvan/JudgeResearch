# Problem 05 Theory - D-FINE Set Criterion

## Core Definitions
- **Set Criterion**: The master loss module that orchestrates all component losses in a DETR-type detector. It receives the matched indices (from Hungarian Matcher), selects aligned prediction-GT pairs, then computes and sums all configured losses.
- **Three Losses in D-FINE**: (1) VFL classification — over all N queries; (2) L1 box — over matched queries only; (3) GIoU regression — over matched queries only. All are normalized by total GT count.
- **D-FINE extension**: The full D-FINE criterion also adds FGL (Fine-Grained Localization) and DDF (Decoupled Distillation Focal) losses using the distribution bins from FDR, and repeats all losses for auxiliary decoder outputs under the GO-LSD union matching. This problem covers the core 3-loss structure.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `pred_logits` | `(B, N, C)` | All query class logits |
| `pred_boxes` | `(B, N, 4)` | All query boxes cxcywh |
| `batch_idx, src_idx` | `(T_total,)` | Permutation indices for matched predictions |
| `matched_src_boxes` | `(T_total, 4)` | Matched predicted boxes |
| `matched_tgt_boxes` | `(T_total, 4)` | Corresponding GT boxes |
| `target_classes` | `(B, N)` | int64 class labels (background=C for unmatched) |
| `gt_score` | `(B, N)` | IoU quality, non-zero only for matched queries |
| `loss_vfl` | scalar | VFL loss averaged over queries, normalized by num_boxes |
| `loss_bbox` | scalar | L1 box loss, normalized by num_boxes |
| `loss_giou` | scalar | 1−GIoU loss, normalized by num_boxes |

## Main Equations (LaTeX)

**L1 box loss:**
$$ \mathcal{L}_{bbox} = \frac{1}{N_{gt}} \sum_{i \in \text{matched}} \| \hat{b}_i - b_i^{GT} \|_1 $$

**GIoU regression loss:**
$$ \mathcal{L}_{giou} = \frac{1}{N_{gt}} \sum_{i \in \text{matched}} (1 - \text{GIoU}(\hat{b}_i, b_i^{GT})) $$

**VFL classification loss (full N queries):**
$$ \mathcal{L}_{vfl} = \frac{N}{N_{gt}} \cdot \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} w_{ic} \cdot \text{BCE}(\hat{l}_{ic}, q_{ic}) $$

where $q_{ic} = \text{IoU}(\hat{b}_i, b_i^{GT})$ if $c = \text{label}(i)$ and matched, else 0.

## Step-by-Step Derivation or Computation Flow
1. Build `(batch_idx, src_idx)` from `indices` via `_get_src_permutation_idx`.
2. Gather `matched_src = pred_boxes[batch_idx, src_idx]` → `(T, 4)`.
3. Gather `matched_tgt = cat([t['boxes'][j]...])` → `(T, 4)`.
4. `loss_bbox = F.l1_loss(matched_src, matched_tgt, reduction='sum') / num_boxes`.
5. Convert to xyxy. `giou = diag(GIoU(src_xyxy, tgt_xyxy))`. `loss_giou = (1-giou).sum() / num_boxes`.
6. Fill `target_classes(B,N)` = `num_classes`. Set `target_classes[batch_idx, src_idx] = labels`.
7. Compute IoU between matched boxes → `ious` `(T,)`. Build `gt_score(B,N)` with ious at matched.
8. Flatten logits `(B*N, C)`, labels and scores. Call `varifocal_loss`.
9. `loss_vfl = vfl.sum() / (num_boxes * N)` (normalize per query then scale by N/num_boxes).

## Tensor Shape Flow
```
pred_logits: (B, N, C)              pred_boxes: (B, N, 4)
             ↓ index [batch_idx, src_idx]
matched_src: (T_total, 4) ←→ matched_tgt: (T_total, 4)
             ↓ L1                         ↓ GIoU
loss_bbox: scalar            loss_giou: scalar

pred_logits: (B, N, C)        target_classes: (B, N)   gt_score: (B, N)
             ↓ flatten
(B*N, C)  + (B*N,) labels + (B*N,) scores → VFL → sum/num_boxes → loss_vfl
```

## Practical Interpretation
The Set Criterion is deliberately "dumb" — it does not know anything about the matching procedure. It just receives indices and faithfully computes losses between selected pairs. This decoupling makes the loss orthogonal to matching strategy changes. In D-FINE's full implementation, the same criterion is called multiple times (once per decoder layer, once for the encoder proposal head) with different indices, enabling multi-layer supervision without architecture changes.
