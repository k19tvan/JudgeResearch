# Problem 05 - D-FINE Set Criterion

## Description
`SetCriterion` is the centralized loss module in D-FINE. Given matched indices from the Hungarian Matcher, it aggregates three losses: VFL classification loss, L1 bounding box regression, and GIoU regression. It also orchestrates auxiliary losses from intermediate decoder layers and encoder heads. Your task is to implement a simplified `SetCriterion` that computes these three losses for the final decoder output only (no auxiliary outputs, no FDR — those are Problem 11's assembly step).

## Input Format
`outputs`: dict with:
- `pred_logits`: `(B, N, C)`, float32.
- `pred_boxes`:  `(B, N, 4)`, float32, normalized `cxcywh`.

`targets`: list of B dicts, each with:
- `labels`: `(T_i,)`, int64.
- `boxes`: `(T_i, 4)`, float32, normalized `cxcywh`.

`indices`: list of B tuples `(src_idx, tgt_idx)` from HungarianMatcher.

## Output Format
Dict with scalar losses:
- `loss_vfl`:  float32 scalar.
- `loss_bbox`: float32 scalar (L1).
- `loss_giou`: float32 scalar.

## Constraints
- All losses normalized by total `num_boxes = sum(T_i)` across the batch.
- `pred_boxes` and `target_boxes` are in `cxcywh` — convert to `xyxy` only for GIoU.
- Only matched predictions (selected by `src_idx`) incur box regression losses.
- VFL loss is computed over ALL N queries per image (unmatched query class = background).

## Example
```python
# B=1, N=5, C=3, T=2
outputs = {"pred_logits": torch.randn(1, 5, 3), "pred_boxes": torch.rand(1, 5, 4)}
targets = [{"labels": torch.tensor([0, 2]), "boxes": torch.rand(2, 4)}]
indices = [(torch.tensor([1, 3]), torch.tensor([0, 1]))]  # from matcher
losses = criterion(outputs, targets, indices)
# losses = {"loss_vfl": tensor(...), "loss_bbox": tensor(...), "loss_giou": tensor(...)}
```

## Hints
- Use `_get_src_permutation_idx(indices)` to produce `(batch_idx, src_idx)` for advanced indexing.
- `matched_src_boxes  = outputs['pred_boxes'][batch_idx, src_idx]` → `(T_total, 4)`.
- `matched_tgt_boxes  = cat([t['boxes'][j] for t,(_, j) in zip(targets, indices)])`.
- `loss_bbox = F.l1_loss(matched_src, matched_tgt) / num_boxes`.
- `loss_giou = (1 - diag(GIoU(xyxy_src, xyxy_tgt))).sum() / num_boxes`.
- For VFL: fill a label matrix `(B, N)` with `num_classes` (background), then set matched positions to their class.
- Build IoU quality for matched queries, then call `varifocal_loss`.

## Checker
```bash
python checker.py
```
