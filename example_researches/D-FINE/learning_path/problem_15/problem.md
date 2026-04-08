# Problem 15: Multi-Layer Training with D-FINE Criterion

## Problem Statement

Now that you understand:
- ✓ **Matching Union (Problem 13)**: Consensus matching across decoder/encoder layers
- ✓ **FGL Loss (Problem 14)**: Distribution-based bounding box regression

You'll combine these components into a **complete D-FINE training criterion** that:

1. Performs matching internally (no external matcher needed)
2. Computes consensus matching union from auxiliary layers
3. Applies multi-layer loss supervision
4. Aggregates losses from all decoder layers
5. Handles gradient flow for end-to-end training

## Context

In D-FINE, the model produces predictions from **every decoder layer**, not just the final layer. This provides two benefits:

1. **Intermediate supervision**: Encourage all layers to produce good predictions
2. **Consensus matching**: Use multiple perspectives for better assignment quality

The **criterion** (loss function) is responsible for:
- Taking model outputs (including auxiliary outputs)
- Taking ground truth targets
- Internally calling the matcher multiple times
- Computing expected metric losses (VFL, FGL, IoU)
- Aggregating across all layers
- Returning total weighted loss

## Files to Fetch

Look at the existing files in `dfine_mini/`:
- `criterion.py` - Current single-layer implementation
- `matcher.py` - Hungarian matcher with `compute_matching_union()` method
- `losses.py` - VFL loss implementation
- `fgl_loss.py` - FGL (Fine-Grained Localization) loss

Study how `matcher` and `losses` work before implementing the criterion.

## Your Task

Implement a **paper-compliant D-FINE criterion** that:

### (A) Constructor (`__init__)
```python
class DFINECriterion(nn.Module):
    def __init__(self, 
                 matcher: HungarianMatcher,
                 num_classes: int,
                 weight_dict: dict,
                 losses: list,
                 num_layers: int = 6,
                 reg_max: int = 32):
        """
        Args:
            matcher: HungarianMatcher with compute_matching_union()
            num_classes: number of detection classes
            weight_dict: {"loss_vfl": w1, "loss_fgl": w2, ...}
            losses: ["vfl", "fgl", ...] - which losses to compute
            num_layers: number of decoder layers (for multi-layer loop)
            reg_max: bins for FGL loss discretization
        """
```

### (B) Core Forward Logic
```python
def forward(self, outputs, targets):
    """
    Args:
        outputs: {
            "pred_logits": (B, N, C),
            "pred_boxes": (B, N, 4),
            "pred_corners": (B, N, 4*(reg_max+1)),
            "aux_outputs": [
                {"pred_logits": ..., "pred_boxes": ..., ...},
                ...  # For each decoder layer (except final)
            ]
        }
        targets: list of dicts with keys:
            - "labels": (M,) class labels
            - "boxes": (M, 4) normalized boxes [0,1]
            - "image_id": int identifying original image
    
    Returns:
        losses: dict with {
            "loss_vfl": scalar,
            "loss_fgl": scalar,
            "loss_vfl_aux_0": scalar,  # Aux layer 0 loss
            ...
        }
    """
```

### (C) Key Implementation Steps

**Step 1:** Match final layer predictions with targets
```python
indices_final = self.matcher(outputs, targets)
```

**Step 2:** Match each auxiliary layer, collect indices
```python
indices_aux_list = []
for aux_output in outputs.get("aux_outputs", []):
    indices_aux = self.matcher(aux_output, targets)
    indices_aux_list.append(indices_aux)
```

**Step 3:** Compute matching union (consensus matching)
```python
indices_go = self.matcher.compute_matching_union(indices_aux_list)
# Use this for box regression where consensus matters
```

**Step 4:** Compute losses for final layer
```python
losses = {}
losses.update(self.get_loss(outputs, targets, indices_final, prefix=""))
```

**Step 5:** Compute losses for each auxiliary layer
```python
for i, aux_output in enumerate(outputs.get("aux_outputs", [])):
    losses.update(
        self.get_loss(aux_output, targets, indices_aux_list[i], 
                     prefix=f"aux_{i}_")
    )
```

### (D) Loss Computation per Layer
```python
def get_loss(self, outputs, targets, indices, prefix=""):
    """
    Compute losses for a single layer.
    
    For each loss type in self.losses:
    1. Extract predictions and ground truth using indices
    2. Compute loss (VFL, FGL, or other)
    3. Apply weighting from self.weight_dict
    4. Add to dict with prefix
    
    Returns:
        layer_losses: {"loss_vfl": scalar, "loss_fgl": scalar, ...}
    """
```

**Sub-step D1**: VFL Loss (Classification)
```python
if "vfl" in self.losses:
    pred_logits = outputs["pred_logits"]  # (B, N, C)
    
    # Extract positive/negative sample logits using indices
    # For each matched (query, target) pair:
    #   - Extract pred_logits[batch, query_idx, :]
    #   - Create target score (1.0 for positive, 0.0 for negative)
    
    loss_vfl = varifocal_loss(pred_logits_positive, scores_positive)
    losses[f"{prefix}loss_vfl"] = loss_vfl * self.weight_dict["loss_vfl"]
```

**Sub-step D2**: FGL Loss (Localization)
```python
if "fgl" in self.losses:
    pred_boxes = outputs["pred_boxes"]      # (B, N, 4)
    pred_corners = outputs["pred_corners"]   # (B, N, 4*(reg_max+1))
    
    # Denormalize predictions to image coordinates
    # Extract matched boxes and corners
    # Compute FGL loss using bbox2distance + focal loss
    
    loss_fgl = fgl_loss_result
    losses[f"{prefix}loss_fgl"] = loss_fgl * self.weight_dict["loss_fgl"]
```

## Expected Behavior

1. **Input shape handling**: Process batches with different numbers of objects
2. **Multi-layer flow**: Unroll all decoder layers in order
3. **Matching strategy**: Use final layer matching for final loss, auxiliary matching for each layer
4. **Weight aggregation**: Sum weighted losses from all layers
5. **Gradient flow**: All parameters retain gradients for backprop

## Example Usage

```python
matcher = HungarianMatcher(...)
criterion = DFINECriterion(
    matcher=matcher,
    num_classes=80,
    weight_dict={
        "loss_vfl": 1.0,
        "loss_fgl": 1.0,
    },
    losses=["vfl", "fgl"],
    num_layers=6,
    reg_max=32
)

# Training loop
for images, targets in dataloader:
    outputs = model(images)  # Has aux_outputs
    losses = criterion(outputs, targets)
    loss = losses["loss_vfl"] + losses["loss_fgl"] + ...
    loss.backward()
```

## Hints

1. **Indices unpacking**: `indices` is a list of (pred_idx, target_idx) tuples per batch
2. **Slicing patterns**: Use advanced indexing to extract matched predictions/targets
3. **Tensor stacking**: Handle variable-length matches (different objects per image)
4. **Normalization**: Box coordinates are typically in [0, 1], may need denormalization
5. **Loss aggregation**: Use `sum()` for numerator, accumulate denominators separately

## Validation

Your implementation should:
- ✓ Accept multi-layer outputs with aux_outputs
- ✓ Internally call matcher (not passed externally)
- ✓ Compute matching union consensus
- ✓ Produce dict with loss_vfl, loss_fgl, aux_{i}_loss_vfl, aux_{i}_loss_fgl
- ✓ All losses should be finite (no NaN/Inf)
- ✓ Gradients flow to model parameters

## References

- [D-FINE Paper](https://arxiv.org/abs/2410.13842): Section 3 (Method) details multi-layer supervision
- [DETR Paper](https://arxiv.org/abs/2005.12136): Criterion design pattern
- `matcher.py`: Reference `compute_matching_union()` implementation
- `fgl_loss.py`: Reference FGL loss computation
- `losses.py`: VFL loss implementation
