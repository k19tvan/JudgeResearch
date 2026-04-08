# Question 15: Multi-Layer D-FINE Criterion Implementation

## Task

Implement `DFINECriterion` that combines:
1. **Matcher calls** - Internal matching for each layer
2. **Matching union** - Consensus indices across layers
3. **Multi-layer losses** - VFL (classification) + FGL (localization)
4. **Loss aggregation** - Sum losses from all decoder layers

## Skeleton Code

```python
import torch
import torch.nn as nn
from matcher import HungarianMatcher, compute_matching_union
from losses import varifocal_loss
from fgl_loss import bbox2distance, unimodal_distribution_focal_loss


class DFINECriterion(nn.Module):
    """D-FINE Criterion with multi-layer supervision and matching consensus."""
    
    def __init__(self, 
                 matcher: HungarianMatcher,
                 num_classes: int,
                 weight_dict: dict,
                 losses: list,
                 num_layers: int = 6,
                 reg_max: int = 32):
        """
        Args:
            matcher: HungarianMatcher instance
            num_classes: Number of object classes
            weight_dict: Loss weights {"loss_vfl": 1.0, "loss_fgl": 1.0, ...}
            losses: List of loss types to compute ["vfl", "fgl"]
            num_layers: Number of decoder layers
            reg_max: Bins for FGL distribution (32 means 0-32 inclusive)
        """
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_layers = num_layers
        self.reg_max = reg_max
    
    def forward(self, outputs, targets):
        """
        Compute D-FINE loss with multi-layer supervision.
        
        Args:
            outputs: Model output dictionary with:
                - pred_logits: (B, N, num_classes) final layer logits
                - pred_boxes: (B, N, 4) final layer boxes
                - pred_corners: (B, N, 4*(reg_max+1)) FGL predictions
                - aux_outputs: list of aux layer outputs (same structure)
            
            targets: List[Dict] with:
                - labels: (M,) class labels for this image
                - boxes: (M, 4) normalized ground truth boxes
                - image_id: int image identifier
        
        Returns:
            losses: Dict[str, Tensor] with loss values
                - "loss_vfl": final layer VFL loss
                - "loss_fgl": final layer FGL loss
                - "aux_0_loss_vfl": layer 0 (first aux) VFL loss
                - "aux_0_loss_fgl": layer 0 FGL loss
                - ... (for each auxiliary layer)
        """
        # TODO: Implement
        losses = {}
        
        # Step 1: Match final layer
        # indices_final = ...
        
        # Step 2: Match all auxiliary layers
        # indices_aux_list = [self.matcher(...) for aux_out in aux_outputs]
        
        # Step 3: Compute matching union
        # indices_go = self.matcher.compute_matching_union(indices_aux_list)
        
        # Step 4: Compute final layer losses
        # losses.update(self.get_loss(outputs, targets, indices_final, prefix=""))
        
        # Step 5: Compute auxiliary layer losses
        # for i, aux_output in enumerate(outputs.get("aux_outputs", [])):
        #     losses.update(...)
        
        return losses
    
    def get_loss(self, outputs, targets, indices, prefix=""):
        """
        Compute losses for a single layer.
        
        Args:
            outputs: Dict with pred_logits, pred_boxes, pred_corners, etc.
            targets: List[Dict] ground truth
            indices: List[Tuple[Tensor, Tensor]] - matched indices per image
            prefix: String prefix for loss keys (e.g., "aux_0_", "")
        
        Returns:
            losses: Dict with weighted loss values
        """
        layer_losses = {}
        
        # TODO: Compute VFL loss if "vfl" in self.losses
        
        # TODO: Compute FGL loss if "fgl" in self.losses
        
        return layer_losses
```

## Implementation Details

### Step 1: Extract Indices

For each batch sample `b` and each (src_idx, tgt_idx) pair, extract:
- Prediction: `outputs["pred_logits"][b, src_idx, :]`
- Target: Get label from `targets[b]["labels"][tgt_idx]`

### Step 2: VFL Loss

```python
# Collect matched predictions and targets
pred_logits_matched = []  # Shape: (total_matched, num_classes)
target_labels = []        # Shape: (total_matched,)

for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
    pred = outputs["pred_logits"][batch_idx, src_indices]  # (num_matched, C)
    tgt = targets[batch_idx]["labels"][tgt_indices]       # (num_matched,)
    
    pred_logits_matched.append(pred)
    target_labels.append(tgt)

pred_logits_matched = torch.cat(pred_logits_matched)  # (total_matched, C)
target_labels = torch.cat(target_labels)              # (total_matched,)

# Create target scores for VFL
target_scores = torch.zeros_like(pred_logits_matched)
target_scores[range(len(target_labels)), target_labels] = 1.0

loss_vfl = varifocal_loss(pred_logits_matched, target_scores)
```

### Step 3: FGL Loss

```python
# Extract matched boxes and corners
pred_boxes_matched = []    # (total_matched, 4)
pred_corners_matched = []  # (total_matched, 4*(reg_max+1))
target_boxes = []          # (total_matched, 4)

for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
    pred_box = outputs["pred_boxes"][batch_idx, src_indices]      # (num_matched, 4)
    pred_corner = outputs["pred_corners"][batch_idx, src_indices] # (num_matched, 4*(reg_max+1))
    tgt_box = targets[batch_idx]["boxes"][tgt_indices]           # (num_matched, 4)
    
    pred_boxes_matched.append(pred_box)
    pred_corners_matched.append(pred_corner)
    target_boxes.append(tgt_box)

# Compute FGL loss
# Use bbox2distance to convert target boxes to soft labels
# Then compute focal loss between pred_corners and soft labels
```

### Step 4: Loss Aggregation

```python
layer_losses[f"{prefix}loss_vfl"] = loss_vfl * self.weight_dict.get("loss_vfl", 1.0)
layer_losses[f"{prefix}loss_fgl"] = loss_fgl * self.weight_dict.get("loss_fgl", 1.0)
```

## Test Cases

Your implementation should handle:

1. **Single image batch**: B=1 with variable objects
2. **Multi-image batch**: B>1 with different object counts
3. **Empty images**: Some images with 0 objects
4. **All auxiliary layers**: Process all 5 aux layers (6 total layers)
5. **Consistent shapes**: Output dict has correct loss names and shapes

## Expected Output

```python
outputs = {
    "pred_logits": torch.randn(2, 100, 80),
    "pred_boxes": torch.sigmoid(torch.randn(2, 100, 4)),
    "pred_corners": torch.randn(2, 100, 4*33),
    "aux_outputs": [
        {
            "pred_logits": torch.randn(2, 100, 80),
            "pred_boxes": torch.sigmoid(torch.randn(2, 100, 4)),
            "pred_corners": torch.randn(2, 100, 4*33),
        }
        for _ in range(5)  # 5 auxiliary layers
    ]
}

losses = criterion(outputs, targets)

# Expected keys:
# - "loss_vfl", "loss_fgl"
# - "aux_0_loss_vfl", "aux_0_loss_fgl"
# - ... (for each aux layer)
# All values: scalar tensors (shape (), dtype float32)
```

## Gradient Flow

Ensure:
- ✓ All loss values are differentiable (require_grad=True)
- ✓ Losses can be summed: `total_loss = sum(losses.values())`
- ✓ `.backward()` propagates gradients to model parameters
- ✓ No in-place operations that break gradients

## Common Pitfalls

❌ Forget to set `requires_grad=True` on target tensors
❌ Use in-place operations (`.fill_()`, `.copy_()`) breaking computation graph
❌ Misaligned batch dimensions when stacking predictions
❌ Not handling variable-length matches per image
❌ Forgetting to apply weight_dict scaling to losses
