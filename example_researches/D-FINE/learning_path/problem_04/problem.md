# Problem 04 - Varifocal Loss (VFL) for Classification

## Description
D-FINE uses Varifocal Loss (VFL) for classification — a variant of Focal Loss that jointly encodes both the predicted quality (IoU with matched ground truth) and the class probability into a single soft target. Unlike standard cross-entropy or Focal Loss which uses binary 0/1 labels, VFL sets the positive target score to the actual IoU between the predicted and ground-truth box. This aligns the confidence score with localization quality. Your task is to implement `varifocal_loss` and the simpler baseline `sigmoid_focal_loss` both from scratch.

## Input Format
`sigmoid_focal_loss(inputs, targets, alpha, gamma)`:
- `inputs`: `(B, N, C)` or `(N, C)`, float32 — raw logits.
- `targets`: same shape as inputs, float32 — binary targets in `{0, 1}`.
- `alpha`: float, default 0.25.
- `gamma`: float, default 2.0.
Returns: scalar loss.

`varifocal_loss(pred_logit, gt_score, label, alpha, gamma)`:
- `pred_logit`: `(N, C)`, float32 — raw logits.
- `gt_score`: `(N,)`, float32 — IoU-based quality scores in `[0,1]`.
- `label`: `(N,)`, int64 — ground truth class index.
- `alpha`: float, default 0.75.
- `gamma`: float, default 2.0.
Returns: `(N, C)` elementwise loss tensor.

## Output Format
`sigmoid_focal_loss`: scalar float32.
`varifocal_loss`: `(N, C)` float32 tensor.

## Constraints
- No `torchvision.ops.sigmoid_focal_loss` — implement from scratch.
- Numerically stable: use `F.binary_cross_entropy_with_logits` to compute BCE rather than manual `log(sigmoid(...))`.
- `varifocal_loss` target for positive class = `gt_score`, for negative = 0.
- `varifocal_loss` weight: positive = `gt_score`, negative = `alpha * pred_prob^gamma`.

## Example
```python
# N=2, C=3
pred = torch.tensor([[2.0, -1.0, -1.0], [-1.0, 3.0, -1.0]])
gt_score = torch.tensor([0.9, 0.7])   # IoU quality scores
label    = torch.tensor([0, 1])        # class indices
# VFL target for row 0: [0.9, 0, 0]; row 1: [0, 0.7, 0]
```

## Hints
- Build one-hot encoding via `F.one_hot(label, num_classes=C).float()`.
- VFL target: `target = one_hot * gt_score.unsqueeze(-1)`.
- VFL weight: `weight = alpha * sigmoid(pred_logit).pow(gamma) * (1 - one_hot) + one_hot * gt_score.unsqueeze(-1)`.
- Apply `F.binary_cross_entropy_with_logits(pred_logit, target, weight=weight, reduction='none')`.
- Focal weight for `sigmoid_focal_loss`: `p_t = p * t + (1-p)*(1-t)` where `p=sigmoid(inputs), t=targets`.

## Checker
```bash
python checker.py
```
