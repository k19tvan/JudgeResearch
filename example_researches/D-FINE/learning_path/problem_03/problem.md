# Problem 03 - Hungarian Matcher (Bipartite Matching)

## Description
DETR-family detectors make N independent predictions per image (typically N=300) and must match them 1-to-1 with ground truth boxes to compute meaningful losses. The Hungarian Algorithm solves this optimal bipartite matching by minimizing a combined cost (classification + L1 box + GIoU). Your task is to implement `HungarianMatcher` — a `torch.no_grad()` module that computes a cost matrix and returns the optimal assignment of predictions to ground-truth boxes.

## Input Format
`outputs`: dict with:
- `pred_logits`: `(B, N, C)`, float32 — raw classification scores.
- `pred_boxes`: `(B, N, 4)`, float32 — predicted boxes in `cxcywh`, normalized.

`targets`: list of B dicts, each containing:
- `labels`: `(T_i,)`, int64 — ground truth class ids.
- `boxes`: `(T_i, 4)`, float32 — ground truth boxes in `cxcywh`, normalized.

where `T_i` = number of ground-truth objects in image `i`.

## Output Format
`indices`: list of B tuples `(src_idx, tgt_idx)`, each being int64 tensors of length `min(N, T_i)`.

Semantics: `pred_boxes[b, src_idx[k]]` is matched to `targets[b]['boxes'][tgt_idx[k]]`.

## Constraints
- N >> T_i in practice (300 queries, 1-50 targets). Most queries are unmatched.
- `linear_sum_assignment` (from `scipy.optimize`) must be called on CPU tensors.
- All cost computations must be done in batch first, then split per image.
- `cost_class = focal_cost(pred_prob, tgt_ids)` with alpha=0.25, gamma=2.0.

## Example
```python
# B=1, N=3, C=2, T=2
pred_logits = torch.randn(1, 3, 2)
pred_boxes  = torch.rand(1, 3, 4)  # cxcywh
targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4)}]
# Output: [(src_idx, tgt_idx)] each of length 2
```

## Hints
- Flatten pred to `(B*N, C)` and `(B*N, 4)` before cost computation.
- Focal classification cost: `pos = alpha*(1-p)^gamma * (-log(p+eps))`; `neg = (1-alpha)*p^gamma * (-log(1-p+eps))`; `cost_class = pos - neg` at target class columns.
- `torch.cdist(pred_boxes_flat, tgt_bbox_flat, p=1)` gives L1 cost.
- Use `generalized_box_iou` from Problem 02 for GIoU cost.
- Final cost: `C = w_bbox * cost_L1 + w_class * cost_class + w_giou * cost_giou`, shaped `(B, N, T_total)`.
- Split cost per image by `sizes = [T_i for each image]`.
- Call `scipy.optimize.linear_sum_assignment(c)` for each sub-cost matrix.

## Checker
```bash
python checker.py
```
