"""Problem 14: Solution - Fine-Grained Localization Loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox2distance(points, bbox, reg_max=32, stride=1):
    \"\"\"Convert bounding boxes to distance distributions from reference points.
    
    Args:
        points: (N, 2) Reference points [x, y]
        bbox: (N, 4) Bounding boxes [x1, y1, x2, y2]
        reg_max: Maximum bin index (reg_max+1 total bins)
        stride: Feature map stride (default 1)
    
    Returns:
        distances: (N, 4) Raw distances [left, right, top, bottom]
        soft_label: (N, 4, reg_max+1) Soft labels for distribution
        weight: (N, 4) Per-distance weight
    \"\"\"
    # Extract reference point coordinates
    x = points[:, 0]  # (N,)
    y = points[:, 1]  # (N,)
    
    # Extract bbox coordinates
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    
    # Compute distances from point to each side
    left = x - x1    # Distance to left edge
    right = x2 - x   # Distance to right edge
    top = y - y1     # Distance to top edge
    bottom = y2 - y  # Distance to bottom edge
    
    # Stack distances: (N, 4)
    distances = torch.stack([left, right, top, bottom], dim=-1)
    
    # Clamp distances to [0, reg_max]
    distances = distances.clamp(min=0, max=reg_max)
    
    # Quantize to bins via linear interpolation
    distances_int = distances.long()  # Lower bin index
    distances_frac = distances - distances_int.float()  # Interpolation weight
    
    # Create soft labels
    soft_label = torch.zeros(
        distances.shape[0], distances.shape[1], reg_max + 1,
        device=distances.device,
        dtype=distances.dtype
    )
    
    # For each distance, distribute probability between adjacent bins
    for i in range(distances.shape[0]):
        for j in range(4):
            bin_lower = distances_int[i, j]
            bin_upper = min(bin_lower + 1, reg_max)
            frac = distances_frac[i, j]
            
            # Lower bin gets (1 - frac) probability
            soft_label[i, j, bin_lower] = 1.0 - frac
            
            # Upper bin gets frac probability (if different)
            if bin_upper < reg_max + 1 and bin_upper != bin_lower:
                soft_label[i, j, bin_upper] = frac
    
    # Compute weights: penalize out-of-range distances
    weight = torch.ones_like(distances)
    # Mark out-of-range with lower weight for gradient
    original_dist = torch.stack([x - x1, x2 - x, y - y1, y2 - y], dim=-1)
    weight[original_dist > reg_max] = 0.5
    weight[original_dist < 0] = 0.5
    
    return distances, soft_label, weight


def unimodal_distribution_focal_loss(
    pred_dist,
    soft_label,
    weight=None,
    alpha=0.25,
    gamma=2.0,
    reduction='none'
):
    \"\"\"Focal loss over distance distributions (unimodal assumption).
    
    Args:
        pred_dist: (N, 4, reg_max+1) Predicted logits for each distance bin
        soft_label: (N, 4, reg_max+1) Target soft labels
        weight: (N, 4) Optional per-distance weight (e.g., from IoU)
        alpha: Focal loss alpha
        gamma: Focal loss gamma
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        loss: Focal loss, shape depends on reduction
    \"\"\"
    # Compute probabilities
    pred_prob = F.softmax(pred_dist, dim=-1)  # (N, 4, reg_max+1)
    
    # KL divergence as main loss (distribution matching)
    kl_loss = F.kl_div(
        F.log_softmax(pred_dist, dim=-1),
        soft_label,
        reduction='none'
    ).sum(dim=-1)  # (N, 4) - sum over bins
    
    # Compute focal term: (1 - p_t)^gamma
    # p_t = probability assigned to target distribution
    p_t = (pred_prob * soft_label).sum(dim=-1)  # (N, 4)
    p_t = torch.clamp(p_t, min=1e-6, max=1.0)  # Numerical stability
    
    # Focal weight
    focal_weight = (1 - p_t) ** gamma
    
    # Combine: alpha weighting + focal weight
    loss = alpha * focal_weight * kl_loss  # (N, 4)
    
    # Apply per-distance weight if provided
    if weight is not None:
        loss = loss * weight
    
    # Apply reduction
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f\"Unknown reduction: {reduction}\")
