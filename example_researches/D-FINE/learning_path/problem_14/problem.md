# Problem 14: Fine-Grained Localization Loss

## Objective
Implement D-FINE's signature **Fine-Grained Localization (FGL) loss**, which represents bounding boxes as **distributions over distance bins** rather than direct coordinates. This is the core innovation that makes D-FINE "fine-grained".

## Why Fine-Grained?

Traditional box regression:
```python
# DETR/Faster R-CNN approach
pred_boxes = linear_regression([x, y, w, h]) → (B, N, 4)
# Output: 4 continuous values per box
```

D-FINE's fine-grained approach:
```python
# Represent each distance as distribution
# For left, right, top, bottom distances from reference point
pred_distribution = [[prob_0, prob_1, ..., prob_reg_max],  # left
                     [prob_0, prob_1, ..., prob_reg_max],  # right
                     [prob_0, prob_1, ..., prob_reg_max],  # top
                     [prob_0, prob_1, ..., prob_reg_max]]  # bottom
# Output: (B, N, 4, reg_max+1)  -- distribution over bins!
```

### Why This Works Better
1. **Scale adaptability**: Different object sizes use different bin ranges
2. **Multimodal handling**: Can represent uncertain predictions with flat distribution
3. **Fine control**: Using (reg_max+1) bins provides fine-grained control (e.g., 33 bins = sub-pixel accuracy)
4. **Focal loss friendly**: Naturally combines with focal loss to emphasize confident peaks

## Key Concepts

### Distance Representation
```
For bounding box B = [x1, y1, x2, y2] and reference point P = [px, py]:
  left_dist    = px - x1   (distance from point to left edge)
  right_dist   = x2 - px   (distance from point to right edge)
  top_dist     = py - y1   (distance from point to top edge)
  bottom_dist  = y2 - py   (distance from point to bottom edge)
  
These 4 distances fully define the box relative to reference point!
```

### Binning (Quantization)
Each distance `d` is converted to soft labels over bins `[0, 1, 2, ..., reg_max]`:
```python
# Example: reg_max=32, distance=5.3
# Maps to bins with soft targets:
#   bin_5: 0.7      (primary)
#   bin_6: 0.3      (interpolation)
# Others: 0.0
```

### Focal Loss over Distributions
```python
# Standard cross-entropy loss
loss_ce = cross_entropy(pred_probs, target_soft_labels)

# Add focal term: (1 - p_t)^gamma
# where p_t = probability assigned to correct bin
loss_focal = focal_weight * loss_ce
```

## Your Task

Implement two main functions in `fgl_loss.py`:

### 1. `bbox2distance(points, bbox, reg_max=32)`
Convert boxes to distance distributions.

**Input**:
- `points`: (N, 2) reference points [x, y]
- `bbox`: (N, 4) boxes in [x1, y1, x2, y2]
- `reg_max`: Number of bins (reg_max+1 total)

**Output**:
- `distances`: (N, 4) actual distances [left, right, top, bottom]
- `soft_label`: (N, 4, reg_max+1) soft labels with interpolation
- `weight`: (N, 4) per-distance weight (e.g., penalize out-of-range)

**Algorithm**:
1. Compute raw distances from point to each side
2. Clamp distances to [0, reg_max]
3. Quantize: distance → two adjacent bins with interpolation
4. Create soft labels via linear interpolation between bins

### 2. `unimodal_distribution_focal_loss(pred_dist, soft_label, weight, ...)`
Compute focal loss over distance distributions.

**Input**:
- `pred_dist`: (N, 4, reg_max+1) predicted distribution logits
- `soft_label`: (N, 4, reg_max+1) target soft labels
- `weight`: (N, 4) optional per-distance weights
- `alpha`, `gamma`: Focal loss parameters

**Output**:
- `loss`: Scalar loss value

**Algorithm**:
1. Softmax to get probabilities
2. Compute KL divergence between prediction and target
3. Weight by focal term: (1 - p_t)^gamma
4. Apply per-distance weights (e.g., IoU-based)
5. Return aggregated loss

## Expected Behavior

Example with 1 query, reg_max=32:
```python
# Box: [50, 60, 150, 200], Reference: [100, 130]
point = [100, 130]
bbox = [50, 60, 150, 200]

# Compute distances
left = 100 - 50 = 50        → bin_50
right = 150 - 100 = 50      → bin_50
top = 130 - 60 = 70         → clamped to 32 (out of range)
bottom = 200 - 130 = 70     → clamped to 32

# Result: distances = [50, 50, 32, 32]
#         soft_label has peaks at these bins
#         weight marks bin_32 as less confident (out-of-range)
```

## Integration

Used in `SetCriterion` when `weight_fgl > 0`:
```python
if "pred_corners" in outputs and weight_fgl > 0:
    # Compute FGL loss on consensus indices
    losses["loss_fgl"] = compute_fgl_loss(outputs, targets, indices_go)
```

## Test Cases

Your implementation should:
1. Handle batch processing (B > 0)
2. Support variable box sizes
3. Interpolate correctly between bins
4. Enforce one-to-one in soft labels
5. Handle out-of-range distances gracefully
6. Maintain numerical stability (no NaN/Inf)

## References
- D-FINE Paper: Section on fine-grained distribution refinement
- Source: `src/zoo/dfine/dfine_utils.py` - `bbox2distance()` implementation
- Source: `src/zoo/dfine/dfine_criterion.py` - FGL loss computation
