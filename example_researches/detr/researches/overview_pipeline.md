# DETR Pipeline Overview (Execution Trace)

This document explains DETR as an actual runtime pipeline: what enters each module, how tensors are transformed, and why each transformation is needed.

For formula-heavy theory and symbol-first derivations, see [researches/overview_theory.md](researches/overview_theory.md).

---

## 1. Reading Guide

This file is organized in execution order:
1. Data enters from dataset and augmentations.
2. Batch is collated/padded into `NestedTensor`.
3. Model forward transforms image -> features -> sequence -> queries -> predictions.
4. Training step computes matching, losses, gradients, and optimizer update.

Conventions used below:
- $B$: batch size.
- $H_i, W_i$: original image size for sample $i$.
- $H_{pad}, W_{pad}$: max height/width in a batch after padding.
- $C$: channel dimension.
- $d$: transformer hidden size (`--hidden_dim`, default 256).
- $N_q$: number of object queries (`--num_queries`, default 100).
- $L$: flattened spatial length $H'W'$.

---

## 2. End-to-End Runtime Map

```text
Image+Annotation
  -> datasets/coco.py: CocoDetection.__getitem__
  -> datasets/transforms.py: geometric + normalize transforms
  -> util/misc.py: collate_fn -> nested_tensor_from_tensor_list
  -> models/detr.py: DETR.forward
      backbone -> position encoding -> input_proj -> transformer
      -> class head + box head (+ optional mask head)
  -> models/matcher.py: HungarianMatcher
  -> models/detr.py: SetCriterion (loss dictionary)
  -> engine.py: reduce/log losses, backward, clip, optimizer.step
```

---

## 3. Data Pipeline: Exact Input Transformations

### 3.1 Dataset Output Before Transforms

Source: [datasets/coco.py](datasets/coco.py)

Per sample, `CocoDetection.__getitem__` starts from:
- PIL image: shape `(H_i, W_i, 3)` in image-space.
- raw COCO annotations list.

Then `ConvertCocoPolysToMask` creates target tensors:
- `target["boxes"]`: absolute `xyxy`, shape `(N_gt_i, 4)`.
- `target["labels"]`: class ids, shape `(N_gt_i,)`.
- `target["image_id"]`: shape `(1,)`.
- `target["orig_size"]`: `[H_i, W_i]`.
- `target["size"]`: current size after transform stage.
- `target["masks"]` optional: `(N_gt_i, H_i, W_i)`.

Meaning:
- At this stage boxes are still absolute pixel coordinates (not normalized).

### 3.2 Geometric Augmentations: Image and Box Stay Aligned

Source: [datasets/transforms.py](datasets/transforms.py)

Typical train pipeline behavior:
- Random resize / crop / horizontal flip.
- Every spatial transform is applied consistently to:
  - image pixels,
  - `target["boxes"]`,
  - `target["masks"]` if present,
  - `target["size"]`.

Important transformation meaning:
- `resize`: scales boxes by width/height ratios.
- `crop`: shifts and clips boxes, removes invalid boxes.
- `hflip`: remaps x coordinates relative to image width.

### 3.3 Normalize Step: Boxes Become DETR Target Format

`Normalize.__call__` does two key things:
1. Normalizes image channels using ImageNet mean/std.
2. Converts boxes from absolute `xyxy` to normalized `cxcywh`:
   - shape remains `(N_gt_i, 4)`
   - coordinates become relative to current transformed image size in `[0,1]`.

This is the exact target space used by DETR box head and box losses.

### 3.4 Collate + Padding: Variable-Size Images -> Batch Tensor

Source: [util/misc.py](util/misc.py)

`collate_fn` calls `nested_tensor_from_tensor_list`:
- Input list: `B` image tensors, each `(3, H_i, W_i)`.
- Output `NestedTensor`:
  - `tensors`: `(B, 3, H_pad, W_pad)` (zero-padded to max in batch).
  - `mask`: `(B, H_pad, W_pad)` bool.

Mask semantics:
- `False` = real image region.
- `True` = padded region.

Why this matters:
- Transformer attention later uses the mask to ignore padded tokens.

---

## 4. Forward Pipeline: Module-by-Module Tensor Trace

Source main entry: [models/detr.py](models/detr.py)

### Step A: DETR.forward Input Contract

- `samples.tensors`: `(B, 3, H_pad, W_pad)`
- `samples.mask`: `(B, H_pad, W_pad)`

If plain list/tensor is passed, it is converted to `NestedTensor` first.

### Step B: Backbone + Position Encoding

Source: [models/backbone.py](models/backbone.py), [models/position_encoding.py](models/position_encoding.py)

`features, pos = backbone(samples)` returns lists over feature levels.

For default non-intermediate usage, DETR takes last level:
- `src`: `(B, C_backbone, H', W')` where `C_backbone=2048` for ResNet50 C5.
- `mask`: `(B, H', W')` (interpolated from input mask).
- `pos[-1]`: `(B, d, H', W')` with `d=256`.

Meaning:
- `src` carries visual semantics.
- `pos` carries spatial identity.
- `mask` marks invalid padded coordinates after downsampling.

### Step C: Channel Projection for Transformer Width

`input_proj = Conv2d(C_backbone -> d, kernel=1)`:
- `src_proj = input_proj(src)`
- shape: `(B, d, H', W')`.

Meaning:
- Aligns backbone channels with transformer hidden size.

### Step D: Transformer Flattening and Encoding

Source: [models/transformer.py](models/transformer.py)

Inside `Transformer.forward`:
- `src_proj`: `(B,d,H',W') -> (L,B,d)` with `L=H'W'`.
- `pos`: `(B,d,H',W') -> (L,B,d)`.
- `mask`: `(B,H',W') -> (B,L)`.

Encoder output:
- `memory`: `(L,B,d)` then reshaped helper form `(B,d,H',W')` for optional use.

Meaning:
- Each of `L` positions is now a token attending globally across the image.

### Step E: Decoder with Learned Object Queries

- Query embedding parameter: `(N_q,d)`.
- Expanded to `(N_q,B,d)`.
- Initial decoder target `tgt`: zeros, same shape `(N_q,B,d)`.

Decoder returns intermediate stack `hs`:
- internal shape `(num_decoder_layers, N_q, B, d)`.
- returned to DETR as `(num_decoder_layers, B, N_q, d)`.

Meaning:
- Each query slot becomes an object hypothesis refined layer by layer.

### Step F: Prediction Heads

From `hs`:
- Class head `Linear(d, num_classes+1)` -> `(num_layers, B, N_q, K+1)`.
- Box head `MLP(d,d,4)` + sigmoid -> `(num_layers, B, N_q, 4)`.

Final output (last decoder layer):
- `pred_logits`: `(B, N_q, K+1)`.
- `pred_boxes`: `(B, N_q, 4)` normalized `cxcywh`.
- `aux_outputs` optional list for earlier decoder layers.

---

## 5. Training Step: Detailed Execution Flow

Source orchestration: [engine.py](engine.py), [models/detr.py](models/detr.py), [models/matcher.py](models/matcher.py)

This section describes one iteration in `train_one_epoch`.

### 5.1 Mini-batch Preparation

From dataloader:
- `samples`: `NestedTensor`.
- `targets`: list length `B`, each with normalized `boxes` and class `labels`.

Move to device:
- `samples = samples.to(device)`.
- each target tensor moved to same device.

### 5.2 Forward Pass

`outputs = model(samples)`:
- `pred_logits`: `(B,N_q,K+1)`
- `pred_boxes`: `(B,N_q,4)`
- optional `aux_outputs`.

### 5.3 Hungarian Matching (No Gradient Through Solver)

Inside criterion:
- `outputs_without_aux` is matched first.
- Matcher flattens predictions:
  - probs from logits -> `(B*N_q, K+1)`.
  - boxes -> `(B*N_q, 4)`.
- Concatenates all GT labels/boxes across batch.
- Builds cost matrix per batch item with class + L1 + GIoU terms.
- Runs `linear_sum_assignment` on CPU for each item.
- Returns per-item `(pred_indices, gt_indices)`.

Meaning:
- Chooses unique best prediction per GT object.
- Unmatched queries become no-object supervision.

### 5.4 Loss Construction in SetCriterion

Main loss components:
1. `loss_ce`: cross-entropy over all `N_q` queries.
2. `loss_bbox`: L1 over matched pairs only.
3. `loss_giou`: GIoU over matched pairs only.
4. `cardinality_error`: logging metric (no grad).
5. mask losses if segmentation enabled.

Normalization detail:
- `num_boxes = total GT boxes across distributed workers` (clamped min 1).
- box/mask losses are normalized by `num_boxes`.

Auxiliary losses:
- repeated matching/loss for each intermediate decoder output.
- keys are suffixed (`loss_ce_0`, `loss_bbox_0`, ...).

### 5.5 Weighted Sum and Backprop

In [engine.py](engine.py):
- `weight_dict` from criterion config selects and scales active losses.
- `losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)`.

Then optimization:
1. `optimizer.zero_grad()`
2. `losses.backward()`
3. optional `clip_grad_norm_(model.parameters(), max_norm)`
4. `optimizer.step()`

Meaning of clipping:
- Prevents unstable gradient spikes, especially early in training.

### 5.6 Logging and Distributed Reduction

- `reduce_dict` averages each loss across GPUs for consistent logging.
- both scaled and unscaled losses are tracked.
- LR and class error are logged each interval.

---

## 6. End-of-Epoch and Scheduler Behavior

Source: [main.py](main.py)

Per epoch:
1. Call `train_one_epoch`.
2. Call `lr_scheduler.step()` (StepLR at `--lr_drop`).
3. Save checkpoints (`checkpoint.pth`, plus periodic snapshots).
4. Run `evaluate` on validation set.

Optimizer parameter groups:
- non-backbone parameters use `--lr`.
- backbone parameters use lower `--lr_backbone`.

Meaning:
- Lower LR for pretrained backbone preserves useful generic features while transformer/heads adapt faster.

---

## 7. Evaluation and Inference Transformations

### 7.1 Eval Forward

Evaluation in [engine.py](engine.py) runs:
- same model forward and criterion loss computation for metrics.
- postprocessing to COCO format.

### 7.2 PostProcess: Normalized Boxes -> Pixel Boxes

Source: `PostProcess` in [models/detr.py](models/detr.py)

Transformations:
1. `softmax` on class dimension.
2. take best non-background class score/label.
3. convert `cxcywh` -> `xyxy`.
4. scale by original image sizes (`orig_size`) to pixel coordinates.

Output per image:
- `scores`: `(N_q,)`
- `labels`: `(N_q,)`
- `boxes`: `(N_q,4)` in absolute pixels.

Meaning:
- DETR itself predicts normalized geometry; postprocess restores real-image coordinates for evaluation.

---

## 8. Concrete Single-Batch Example (Mental Model)

Assume batch has two images:
- Image A: `3x640x960`
- Image B: `3x800x800`

After collate:
- `samples.tensors`: `(2,3,800,960)`
- `samples.mask`: `(2,800,960)`

After backbone C5 (stride 32):
- `src`: roughly `(2,2048,25,30)`
- `mask`: `(2,25,30)`

After input projection:
- `(2,256,25,30)`

Flatten for transformer:
- `L=25*30=750`
- sequence tensor `(750,2,256)`

Decoder output stack:
- `(6,2,100,256)`

Predictions:
- logits `(2,100,K+1)`
- boxes `(2,100,4)` normalized.

If GT counts are 7 and 4:
- matcher creates 7 + 4 matched pairs.
- remaining queries are trained as no-object for classification.

---

## 9. Why Each Step Exists (Quick Meaning Table)

| Step | What changes | Why it is necessary |
|---|---|---|
| Geometric transforms | image + boxes/masks coordinates | data diversity while preserving alignment |
| Normalize boxes to `cxcywh` | absolute -> relative targets | stable scale across image sizes |
| Padding + mask | variable image size -> batch tensor | efficient batching + valid attention masking |
| Backbone | pixels -> semantic feature maps | extract object-relevant visual patterns |
| Positional encoding | add location signal | transformer is order-invariant by default |
| Flatten to sequence | 2D map -> tokens | required by transformer attention |
| Object queries | fixed detection slots | direct set prediction without proposals |
| Hungarian matching | dense predictions -> 1:1 pairs | unique supervision, no NMS in training |
| Weighted set losses | classification + geometry objectives | jointly optimize what/where predictions |
| PostProcess | normalized outputs -> pixel boxes | evaluation and visualization in image coordinates |

---

## 10. File-Level Responsibility Map

- Data parsing and GT conversion: [datasets/coco.py](datasets/coco.py)
- Data transforms and box normalization: [datasets/transforms.py](datasets/transforms.py)
- Batch collation/padding/masks: [util/misc.py](util/misc.py)
- Model forward (backbone+transformer+heads): [models/detr.py](models/detr.py)
- Transformer internals: [models/transformer.py](models/transformer.py)
- Matching: [models/matcher.py](models/matcher.py)
- Loss computation: [models/detr.py](models/detr.py)
- Train/eval loop + optimizer step: [engine.py](engine.py)
- Epoch orchestration + scheduler/checkpointing: [main.py](main.py)

---

## 11. Common Misunderstandings (Important)

1. Query embeddings are not class embeddings.
They are learnable slots that become object hypotheses after decoder attention.

2. `pred_boxes` are normalized to transformed image space, not directly final pixel coordinates.
Use postprocess with target sizes to recover absolute coordinates.

3. Not all 100 queries correspond to objects.
Unmatched queries are expected and supervised as no-object.

4. Hungarian assignment is recomputed every iteration.
Matching depends on current predictions and evolves during training.

5. Auxiliary decoder losses are a training stabilizer.
They provide supervision to intermediate decoder layers, not only final output.
