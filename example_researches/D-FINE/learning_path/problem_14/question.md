## Question 14: Fine-Grained Localization Loss Implementation

Implement the FGL (Fine-Grained Localization) loss components for D-FINE's distribution-based box regression.

### Part A: `bbox2distance(points, bbox, reg_max=32, stride=1)`

Convert bounding boxes to distance distributions from reference points.

**Function Signature**:
```python
def bbox2distance(points, bbox, reg_max=32, stride=1):
    """
    Args:
        points: (N, 2) Reference points in [x, y] format
        bbox: (N, 4) Bounding boxes in [x1, y1, x2, y2] format
        reg_max: Maximum bin index (creates reg_max+1 total bins)
        stride: Feature map stride (default 1 for normalized coordinates)
    
    Returns:
        distances: (N, 4) Raw distances [left, right, top, bottom]
        soft_label: (N, 4, reg_max+1) Soft distribution labels
        weight: (N, 4) Per-distance weight (penalize out-of-range)
    """
```

**Expected Behavior**:
- Extract 4 distances from reference point to box edges
- Clamp distances to [0, reg_max] 
- Create soft labels via linear interpolation between adjacent bins
- Mark out-of-range distances with reduced weight

### Part B: `unimodal_distribution_focal_loss(pred_dist, soft_label, weight, alpha, gamma, reduction)`

Compute focal loss over predicted distance distributions.

**Function Signature**:
```python
def unimodal_distribution_focal_loss(
    pred_dist,
    soft_label,
    weight=None,
    alpha=0.25,
    gamma=2.0,
    reduction='none'
):
    """
    Args:
        pred_dist: (N, 4, reg_max+1) Predicted logits for each bin
        soft_label: (N, 4, reg_max+1) Target soft distribution
        weight: (N, 4) Optional per-distance weight
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        loss: Focal loss, shape depends on reduction parameter
    """
```

**Expected Behavior**:
- Compute softmax over bins
- Calculate KL divergence from prediction to target
- Apply focal weighting: weight *= (1 - p_t)^gamma
- Handle edge cases (numerical stability, empty inputs)

### Testing Requirements

Your implementation must pass:
1. **Single sample**: Verify distances and soft labels for one box
2. **Batch processing**: Handle multiple boxes and reference points
3. **Out-of-range handling**: Correctly handle distances > reg_max
4. **Interpolation**: Verify soft labels sum to ~1.0 per distance-directon
5. **Loss computation**: Ensure loss is numeric and no NaNs
6. **Weight application**: Verify IoU weights reduce loss for poorly localized boxes

### Hints

- Linear interpolation between bins creates soft targets, not hard labels
- KL divergence (or cross-entropy) works better than MSE for distributions
- Focal loss helps the model focus on uncertain/hard-to-predict distances
- Out-of-range distances still contribute to loss but with reduced gradient
