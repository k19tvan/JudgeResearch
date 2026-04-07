# Problem 12 - Training and Evaluation Loop

## Description
The final capstone problem: wire everything together into a functional train/eval loop. Your task is to implement `train_one_epoch()` that runs forward pass, computes losses via `SetCriterion`, backpropagates, and steps the optimizer — and `evaluate()` that runs inference and computes a simple detection metric. This loop must run on synthetic (randomly generated) data so no dataset download is required.

## Input Format
`model`: `DFINEMini` instance.
`criterion`: `SetCriterion` instance.
`matcher`: `HungarianMatcher` instance.
`optimizer`: `torch.optim.AdamW`.
`data_loader`: List of synthetic batches from `make_synthetic_batch()`.
`device`: `torch.device`.

## Output Format
`train_one_epoch(...)` → dict with `avg_loss_vfl`, `avg_loss_bbox`, `avg_loss_giou`, all float.
`evaluate(...)` → dict with `mean_iou` (float, average IoU of top-1 prediction per image).

## Constraints
- `train_one_epoch`: model.train() mode, gradient computation, optimizer.zero_grad() → loss.backward() → optimizer.step().
- Total loss = sum of all loss values in the loss dict.
- `evaluate`: model.eval() mode, `torch.no_grad()`.
- `make_synthetic_batch(B, num_gt)` must generate valid images and targets:
  - images: `(B, 3, 256, 256)` random.
  - targets: list of B dicts with random valid `labels` and `boxes` (cxcywh, positive w/h).

## Example
```python
model = DFINEMini(num_classes=10, num_queries=50)
criterion = SetCriterion(num_classes=10)
matcher   = HungarianMatcher()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loader    = [make_synthetic_batch(2, 3) for _ in range(5)]
train_metrics = train_one_epoch(model, criterion, matcher, optimizer, loader, device)
print(train_metrics)
# {'avg_loss_vfl': 1.23, 'avg_loss_bbox': 0.45, 'avg_loss_giou': 0.89}
```

## Hints
- `make_synthetic_batch`: generate `cx,cy~Uniform(0.2,0.8)`, `w,h~Uniform(0.05,0.3)`. Stack to boxes. Labels = `randint(0, num_classes)`.
- In training loop: `out = model(images)`, then `indices = matcher(out, targets)["indices"]`, then `losses = criterion(out, targets, indices)`.
- Total loss = `sum(losses.values())`.
- For eval: find the best matching prediction-gt pair per image using GIoU.
- Guard against empty GT: skip images with T_i=0.

## Checker
```bash
python checker.py
```
