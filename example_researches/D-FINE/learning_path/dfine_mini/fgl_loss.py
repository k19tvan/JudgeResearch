"""Fine-Grained Localization (FGL) Loss for D-FINE.

D-FINE improves bounding box regression via "fine-grained distribution refinement":
- Instead of regressing box coordinates directly, represent boxes as fine-grained
  distributions of distances from reference points
- Each distance (left, right, top, bottom) is binned into (reg_max + 1) classes
- Use focal loss to classify which bin each distance falls into
- Weighted by IoU to emphasize accurate boxes

This enables more precise localization than direct coordinate regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox2distance(points, bbox, reg_max=32, stride=1):
    """Convert bounding boxes to distance distributions from reference points.
    
    For each reference point, compute the distance to each side of the bounding box,
    then represent these distances as distributions over bins [0, reg_max].
    
    Args:
        points: (N, 2) Reference points in image coordinates [x, y]
        bbox: (N, 4) Bounding boxes in [x1, y1, x2, y2] format
        reg_max: Maximum distance bin index (creates reg_max+1 bins)
        stride: Feature map stride (for scaling distances)
    
    Returns:
        distances: (N, 4) Distance to each side: [left, right, top, bottom]
        soft_label: (N, 4, reg_max+1) Soft label for distribution over bins
        weight: (N, 4) Weight for each distance (based on validity)
    """
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
    
    # Convert distances to distribution targets
    # Each distance d is represented as soft labels over bins [0, 1, 2, ..., reg_max]
    # Using linear interpolation between bin centers
    
    # Clamp distances to valid range [0, reg_max]
    distances = distances.clamp(min=0, max=reg_max)  # (N, 4)
    
    # Quantize to bins
    # For distance d in [0, reg_max], find which two bins it falls between
    distances_int = distances.long()  # Lower bin index
    distances_frac = distances - distances_int.float()  # Interpolation weight
    
    # Create soft labels: linear interpolation between two adjacent bins
    soft_label = torch.zeros(
        distances.shape[0], distances.shape[1], reg_max + 1,
        device=distances.device,
        dtype=distances.dtype
    )
    
    # For each distance, distribute probability between two adjacent bins
    for i in range(distances.shape[0]):
        for j in range(4):  # 4 sides
            bin_lower = distances_int[i, j]
            bin_upper = min(bin_lower + 1, reg_max)
            frac = distances_frac[i, j]
            
            soft_label[i, j, bin_lower] = 1.0 - frac
            if bin_upper < reg_max + 1:
                soft_label[i, j, bin_upper] = frac
    
    # Compute weight: penalize out-of-range distances
    # If distance > reg_max, set lower weight
    weight = torch.ones_like(distances)
    weight[distances == reg_max] = 0.5  # Mark out-of-range
    
    return distances, soft_label, weight


def unimodal_distribution_focal_loss(
    pred_dist, 
    soft_label, 
    weight=None, 
    alpha=0.25, 
    gamma=2.0,
    reduction='none'
):
    """Focal loss over distance distributions (unimodal, meaning single peak expected).
    
    Args:
        pred_dist: (N, 4, reg_max+1) Predicted logits for each distance bin
        soft_label: (N, 4, reg_max+1) Target soft labels from bbox2distance
        weight: (N, 4) Per-distance weight (e.g., from IoU)
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        loss: Focal loss over distance distributions, shape depends on reduction
    """
    # Compute probabilities
    pred_prob = F.softmax(pred_dist, dim=-1)  # (N, 4, reg_max+1)
    
    # Cross-entropy loss per bin
    ce_loss = F.kl_div(
        F.log_softmax(pred_dist, dim=-1),
        soft_label,
        reduction='none'
    ).sum(dim=-1)  # (N, 4)
    
    # Compute focal term
    # p_t = probability of true class
    p_t = (pred_prob * soft_label).sum(dim=-1)  # (N, 4)
    p_t = torch.clamp(p_t, min=1e-6, max=1.0)  # Avoid log(0)
    
    # Focal term: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Combine: alpha weighting + focal weight
    loss = alpha * focal_weight * ce_loss  # (N, 4)
    
    # Apply per-distance weight if provided
    if weight is not None:
        loss = loss * weight
    
    # Apply reduction
    if reduction == 'none':
        return loss  # (N, 4)
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class FGLLoss(nn.Module):
    """Fine-Grained Localization Loss module.
    
    Combines:
    1. Distance distribution focal loss (main FGL)
    2. Optional teacher distillation loss
    
    Encourages model to learn fine-grained peak distributions around true distances.
    """
    
    def __init__(self, reg_max=32, alpha=0.25, gamma=2.0, with_teacher_kl=False):
        """
        Args:
            reg_max: Maximum distance bin index
            alpha: Focal loss alpha
            gamma: Focal loss gamma
            with_teacher_kl: Whether to include KL divergence with teacher (advanced)
        """
        super().__init__()
        self.reg_max = reg_max
        self.alpha = alpha
        self.gamma = gamma
        self.with_teacher_kl = with_teacher_kl
    
    def forward(
        self,
        pred_dist,
        target_bbox,
        ref_points,
        iou_weight=None,
        teacher_dist=None,
        temperature=5.0
    ):
        """Compute FGL loss.
        
        Args:
            pred_dist: (N, 4, reg_max+1) Predicted distance distributions
            target_bbox: (N, 4) Target boxes in [x1, y1, x2, y2]
            ref_points: (N, 2) Reference points [x, y]
            iou_weight: (N,) Weight per proposal from IoU with target (optional)
            teacher_dist: (N, 4, reg_max+1) Teacher prediction for distillation (optional)
            temperature: Temperature for KL divergence with teacher
        
        Returns:
            loss: Scalar loss value
        """
        # Convert boxes to distance distributions
        distances, soft_label, dist_weight = bbox2distance(
            ref_points, 
            target_bbox, 
            reg_max=self.reg_max
        )
        
        # Apply IoU-based weighting if provided
        if iou_weight is not None:
            # Broadcast IoU weight to all 4 distances
            dist_weight = dist_weight * iou_weight.unsqueeze(-1)  # (N, 4)
        
        # Compute focal loss over distance distributions
        loss_fgl = unimodal_distribution_focal_loss(
            pred_dist,
            soft_label,
            weight=dist_weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='mean'
        )
        
        total_loss = loss_fgl
        
        # Optional: Add teacher distillation loss
        if self.with_teacher_kl and teacher_dist is not None:
            loss_kl = F.kl_div(
                F.log_softmax(pred_dist / temperature, dim=-1),
                F.softmax(teacher_dist.detach() / temperature, dim=-1),
                reduction='mean'
            ) * (temperature ** 2)
            total_loss = total_loss + loss_kl
        
        return total_loss
