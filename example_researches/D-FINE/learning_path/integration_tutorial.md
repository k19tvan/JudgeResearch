# D-FINE Learning Path — Integration Tutorial

## Phase 3: Assembling the Full Pipeline

This tutorial explains how to take the code you implemented in Problems 01–12 and assemble it into `dfine_mini/`, a runnable training and evaluation project.

---

## Final Project Tree

```
dfine_mini/
├── box_ops.py          ← P01: box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
├── iou.py              ← P02: box_iou, generalized_box_iou
├── matcher.py          ← P03: HungarianMatcher
├── losses.py           ← P04: varifocal_loss, sigmoid_focal_loss
├── criterion.py        ← P05: SetCriterion
├── positional_encoding.py ← P06: PositionEmbeddingSine2D
├── attention.py        ← P07: MultiHeadAttention
├── encoder.py          ← P08: TransformerEncoderLayer
├── decoder.py          ← P09: TransformerDecoderLayer
├── backbone.py         ← P10: HGNetV2Stem
├── model.py            ← P11: DFINEMini
└── train.py            ← P12: train_one_epoch, evaluate, make_synthetic_batch
```

---

## File Assembly Map

| Final File | Source Problem | What to Copy | Why |
|---|---|---|---|
| `box_ops.py` | P01 `solution.py` | `box_cxcywh_to_xyxy`, `box_xyxy_to_cxcywh` | Used by all box geometry modules |
| `iou.py` | P02 `solution.py` | `box_area`, `box_iou`, `generalized_box_iou` | Used by matcher, criterion, eval |
| `matcher.py` | P03 `solution.py` | `HungarianMatcher` class | Imports from `iou.py`, `box_ops.py` |
| `losses.py` | P04 `solution.py` | `sigmoid_focal_loss`, `varifocal_loss` | Used by `criterion.py` |
| `criterion.py` | P05 `solution.py` | `SetCriterion` class | Imports from `iou.py`, `box_ops.py`, `losses.py` |
| `positional_encoding.py` | P06 `solution.py` | `PositionEmbeddingSine2D` | Used by `model.py` |
| `attention.py` | P07 `solution.py` | `MultiHeadAttention` | Used by `encoder.py`, `decoder.py` |
| `encoder.py` | P08 `solution.py` | `TransformerEncoderLayer` | Imports from `attention.py` |
| `decoder.py` | P09 `solution.py` | `TransformerDecoderLayer` | Imports from `attention.py` |
| `backbone.py` | P10 `solution.py` | `cbr`, `HGNetV2Stem` | Used by `model.py` |
| `model.py` | P11 `solution.py` | `DFINEMini` | Imports all above modules |
| `train.py` | P12 `solution.py` | `make_synthetic_batch`, `train_one_epoch`, `evaluate` | Imports `model.py`, `matcher.py`, `criterion.py` |

---

## Merge Order (Follow Strictly)

1. Copy `box_ops.py` first — no dependencies.
2. Copy `iou.py` — add `from box_ops import box_cxcywh_to_xyxy` if needed.
3. Copy `matcher.py` — replace inline `giou`, `cxcywh_to_xyxy` with imports from step 1-2.
4. Copy `losses.py` — standalone, no internal imports.
5. Copy `criterion.py` — import `iou.py`, `box_ops.py`, `losses.py`.
6. Copy `positional_encoding.py`, `attention.py` — standalone.
7. Copy `encoder.py` — import `attention.py`.
8. Copy `decoder.py` — import `attention.py`.
9. Copy `backbone.py` — standalone.
10. Copy `model.py` — import all of the above.
11. Copy `train.py` — import `model.py`, `matcher.py`, `criterion.py`.

---

## Required Imports Header in Each File

```python
# model.py
from backbone import HGNetV2Stem
from positional_encoding import PositionEmbeddingSine2D
from encoder import TransformerEncoderLayer
from decoder import TransformerDecoderLayer

# train.py
from model import DFINEMini
from matcher import HungarianMatcher
from criterion import SetCriterion

# criterion.py
from box_ops import box_cxcywh_to_xyxy
from iou import generalized_box_iou, box_iou
from losses import varifocal_loss
```

---

## Glue Code

Add this block to `train.py` after the imports:

```python
if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 10
    model     = DFINEMini(nc=NUM_CLASSES, nq=50, d=128, nel=2, ndl=3).to(device)
    matcher   = HungarianMatcher()
    criterion = SetCriterion(nc=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Build data loaders
    train_loader = [make_synthetic_batch(4, 3, NUM_CLASSES, device) for _ in range(20)]
    val_loader   = [make_synthetic_batch(2, 3, NUM_CLASSES, device) for _ in range(5)]
    # Train
    for epoch in range(5):
        metrics    = train_one_epoch(model, criterion, matcher, optimizer, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: {metrics} | eval: {val_metrics}")
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | d_model mismatch between modules | Ensure `d_model` is consistent across `model.py` constructor |
| `NaN loss from first iteration` | Random boxes with negative w/h | Clamp `w,h > 0` in `make_synthetic_batch` |
| `linear_sum_assignment` receives NaN | `cost_giou` has NaN from degenerate boxes | Apply `torch.nan_to_num(C, nan=1.0)` before calling scipy |
| `AssertionError: boxes1 has degenerate boxes` | `x2 < x1` after cxcywh→xyxy conversion | Clamp widths/heights: `w,h = w.abs(), h.abs()` |
| `CUDA out of memory` | Batch too large for available VRAM | Reduce `B=2`, `NQ=50`, `d_model=128` |
| `ModuleNotFoundError: scipy` | scipy not installed in env | `pip install scipy` |
| `grad is None` | Called `model.eval()` during backward | Always `model.train()` before backward pass |

---

## Verification Commands

```bash
# 1. Smoke check — model forward pass (< 2s)
conda run -n env python3 -c "
import torch
from model import DFINEMini
m = DFINEMini(nc=10, nq=50, d=128)
o = m(torch.randn(1, 3, 256, 256))
print('Forward OK:', o['pred_logits'].shape, o['pred_boxes'].shape)
"

# 2. Backward pass check (< 5s)
conda run -n env python3 -c "
import torch
from model import DFINEMini
from matcher import HungarianMatcher
from criterion import SetCriterion
m = DFINEMini(nc=10, nq=50, d=128).train()
matcher = HungarianMatcher(); crit = SetCriterion(10)
x = torch.randn(1, 3, 256, 256)
tgt = [{'labels': torch.tensor([2]), 'boxes': torch.rand(1,4).clamp(0.1,0.9)}]
out = m(x); idx = matcher(out, tgt)['indices']; losses = crit(out, tgt, idx)
sum(losses.values()).backward()
print('Backward OK. Losses:', {k: f'{v.item():.4f}' for k,v in losses.items()})
"

# 3. Short train (10 iterations) — loss must be finite
conda run -n env python3 -c "
from train import DFINEMini, HungarianMatcher, SetCriterion, make_synthetic_batch, train_one_epoch
import torch
device = torch.device('cpu')
m = DFINEMini(nc=10, nq=50, d=128).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
loader = [make_synthetic_batch(2, 2, 10, device) for _ in range(10)]
metrics = train_one_epoch(m, SetCriterion(10), HungarianMatcher(), opt, loader, device)
print('Short train OK:', metrics)
"

# 4. Full end-to-end checker (Phase 4 gate)
conda run -n env python3 problem_12/checker.py

# 5. 1 Epoch training + eval
conda run -n env python3 -c "
from train import *
import torch
device = torch.device('cpu')
m     = DFINEMini(nc=10, nq=50, d=128).to(device)
opt   = torch.optim.AdamW(m.parameters(), lr=1e-4)
train_loader = [make_synthetic_batch(4, 3, 10, device) for _ in range(20)]
val_loader   = [make_synthetic_batch(2, 3, 10, device) for _ in range(5)]
metrics = train_one_epoch(m, SetCriterion(10), HungarianMatcher(), opt, train_loader, device)
eval_m  = evaluate(m, val_loader, device)
print('1 Epoch done. Train:', metrics, '| Eval:', eval_m)
"
```

---

## Expected Success Signals

```
Forward OK: torch.Size([1, 50, 10]) torch.Size([1, 50, 4])
Backward OK. Losses: {'loss_vfl': '1.2345', 'loss_bbox': '0.3456', 'loss_giou': '0.9012'}
Short train OK: {'avg_loss_vfl': 1.23, 'avg_loss_bbox': 0.34, 'avg_loss_giou': 0.89}
All Problem 12 checks passed
  Train metrics: {'avg_loss_vfl': 1.xx, 'avg_loss_bbox': 0.xx, 'avg_loss_giou': 0.xx}
  Eval metrics:  {'mean_iou': 0.0x}
=== ALL 12 PROBLEMS VERIFIED SUCCESSFULLY ===
1 Epoch done. Train: {'avg_loss_vfl': ...} | Eval: {'mean_iou': 0.0x}
```

> [!NOTE]
> `mean_iou` will be very low (0.0–0.05) after only 1 epoch on random data. This is expected — detecting real objects requires training on real data for many epochs. The purpose of this integration tutorial is to validate the **pipeline**, not the **performance**.
