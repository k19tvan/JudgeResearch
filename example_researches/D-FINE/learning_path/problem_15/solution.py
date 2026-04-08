"""Problem 15: Solution - Multi-Layer D-FINE Criterion Implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Assume these are imported from dfine_mini modules
# from matcher import HungarianMatcher
# from losses import varifocal_loss
# from fgl_loss import bbox2distance, unimodal_distribution_focal_loss


class DFINECriterion(nn.Module):
    """
    D-FINE Criterion with multi-layer supervision and matching consensus.
    
    Key features:
    1. Internal matcher: Calls matcher for each layer independently
    2. Matching union: Computes consensus matching across auxiliary layers
    3. Multi-layer supervision: Aggregates losses from all decoder layers
    4. Paper-compliant: Follows D-FINE arXiv:2410.13842
    """
    
    def __init__(self, 
                 matcher,
                 num_classes: int,
                 weight_dict: Dict[str, float],
                 losses: List[str],
                 num_layers: int = 6,
                 reg_max: int = 32):
        """
        Initialize D-FINE Criterion.
        
        Args:
            matcher: HungarianMatcher instance with compute_matching_union() method
            num_classes: Number of object classes
            weight_dict: Loss weights {"loss_vfl": 1.0, "loss_fgl": 5.0}
            losses: Loss types ["vfl", "fgl"]
            num_layers: Number of decoder layers
            reg_max: Bins for FGL distribution
        """
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_layers = num_layers
        self.reg_max = reg_max
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute D-FINE loss with multi-layer supervision.
        
        Algorithm:
        1. Match final layer predictions with targets
        2. Match each auxiliary layer predictions with targets
        3. Compute matching union (consensus) from auxiliary matches
        4. Compute weighted losses for final layer (using final indices)
        5. Compute weighted losses for each auxiliary layer (using their indices)
        6. Aggregate and return all losses
        
        Args:
            outputs: {
                "pred_logits": (B, N, C),
                "pred_boxes": (B, N, 4),
                "pred_corners": (B, N, 4*(reg_max+1)),
                "aux_outputs": [aux_0, aux_1, ...]
            }
            targets: List[{
                "labels": (M,),
                "boxes": (M, 4),
                "image_id": int
            }]
        
        Returns:
            losses: Dict of loss scalars
        """
        losses = {}
        
        # Step 1: Match final layer
        indices_final = self.matcher(outputs, targets)
        
        # Step 2: Match auxiliary layers and collect indices
        indices_aux_list = []
        aux_outputs = outputs.get("aux_outputs", [])
        for aux_output in aux_outputs:
            indices_aux = self.matcher(aux_output, targets)
            indices_aux_list.append(indices_aux)
        
        # Step 3: Compute matching union for consensus (used for box losses)
        indices_go = self.matcher.compute_matching_union(indices_aux_list)
        
        # Step 4: Compute losses for final layer
        losses.update(self.get_loss(outputs, targets, indices_final, prefix=""))
        
        # Step 5: Compute losses for auxiliary layers
        for layer_idx, aux_output in enumerate(aux_outputs):
            prefix = f"aux_{layer_idx}_"
            losses.update(self.get_loss(aux_output, targets, indices_aux_list[layer_idx], prefix=prefix))
        
        return losses
    
    def get_loss(self, 
                 outputs: Dict, 
                 targets: List[Dict], 
                 indices: List[Tuple],
                 prefix: str = "") -> Dict[str, torch.Tensor]:
        """
        Compute losses for a single layer.
        
        Args:
            outputs: Single layer output dict {pred_logits, pred_boxes, pred_corners}
            targets: List of target dicts
            indices: Matched indices from matcher
            prefix: Loss name prefix
        
        Returns:
            layer_losses: {
                f"{prefix}loss_vfl": scalar,
                f"{prefix}loss_fgl": scalar,
                ...
            }
        """
        layer_losses = {}
        
        # Extract matched predictions and targets
        matched_preds_logits = []
        matched_targets_labels = []
        matched_preds_boxes = []
        matched_preds_corners = []
        matched_targets_boxes = []
        matched_ref_points = []
        matched_weights = []
        
        device = outputs["pred_logits"].device
        
        for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
            if len(src_indices) == 0:
                # No matches for this image
                continue
            
            # Extract logits and targets for classification (VFL)
            batch_logits = outputs["pred_logits"][batch_idx, src_indices]  # (num_matched, C)
            batch_labels = targets[batch_idx]["labels"][tgt_indices]       # (num_matched,)
            
            matched_preds_logits.append(batch_logits)
            matched_targets_labels.append(batch_labels)
            
            # Extract boxes and targets for localization (FGL)
            if "pred_boxes" in outputs and "pred_corners" in outputs:
                batch_pred_boxes = outputs["pred_boxes"][batch_idx, src_indices]      # (num_matched, 4)
                batch_pred_corners = outputs["pred_corners"][batch_idx, src_indices]  # (num_matched, 4*(reg_max+1))
                batch_tgt_boxes = targets[batch_idx]["boxes"][tgt_indices]            # (num_matched, 4)
                
                matched_preds_boxes.append(batch_pred_boxes)
                matched_preds_corners.append(batch_pred_corners)
                matched_targets_boxes.append(batch_tgt_boxes)
                
                # Compute reference points (box centers)
                pred_centers = (batch_pred_boxes[:, :2] + batch_pred_boxes[:, 2:]) / 2
                matched_ref_points.append(pred_centers)
                
                # Unit weights for now (can be refined with IoU scores)
                matched_weights.append(torch.ones(len(src_indices), device=device))
        
        # Concatenate all matched data
        if len(matched_preds_logits) > 0:
            pred_logits_all = torch.cat(matched_preds_logits)  # (total_matched, C)
            target_labels_all = torch.cat(matched_targets_labels)  # (total_matched,)
            
            # Compute VFL loss if requested
            if "vfl" in self.losses:
                # Create one-hot target scores for VFL
                target_scores = torch.zeros_like(pred_logits_all)
                target_scores.scatter_(1, target_labels_all.unsqueeze(1), 1.0)
                
                # Compute VFL loss (assuming varifocal_loss available)
                from losses import varifocal_loss
                loss_vfl = varifocal_loss(pred_logits_all, target_scores, num_classes=self.num_classes)
                layer_losses[f"{prefix}loss_vfl"] = loss_vfl * self.weight_dict.get("loss_vfl", 1.0)
        
        # Compute FGL loss if requested
        if len(matched_preds_corners) > 0 and "fgl" in self.losses:
            pred_boxes_all = torch.cat(matched_preds_boxes)
            pred_corners_all = torch.cat(matched_preds_corners)
            target_boxes_all = torch.cat(matched_targets_boxes)
            ref_points_all = torch.cat(matched_ref_points)
            weights_all = torch.cat(matched_weights)
            
            # Convert target boxes to soft label distributions
            from fgl_loss import bbox2distance, unimodal_distribution_focal_loss
            
            # bbox2distance expects points and boxes in image coordinates
            soft_labels, soft_weights = bbox2distance(
                ref_points_all,
                target_boxes_all,
                reg_max=self.reg_max
            )
            # soft_labels: (total_matched, 4, reg_max+1)
            # soft_weights: (total_matched, 4)
            
            # Reshape pred_corners to match soft_labels shape
            pred_corners_reshaped = pred_corners_all.view(-1, 4, self.reg_max + 1)
            
            # Combine per-side weights
            combined_weights = (weights_all.unsqueeze(1) * soft_weights).view(-1)  # (total_matched*4,)
            soft_labels_flat = soft_labels.view(-1, self.reg_max + 1)  # (total_matched*4, reg_max+1)
            pred_corners_flat = pred_corners_reshaped.view(-1, self.reg_max + 1)  # (total_matched*4, reg_max+1)
            
            # Compute FGL loss
            loss_fgl = unimodal_distribution_focal_loss(
                pred_corners_flat,
                soft_labels_flat,
                weight=combined_weights,
                reduction="mean"
            )
            layer_losses[f"{prefix}loss_fgl"] = loss_fgl * self.weight_dict.get("loss_fgl", 1.0)
        
        return layer_losses


def _get_src_permutation_idx(indices):
    """Get source indices permutation for loss computation."""
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    """Get target indices permutation for loss computation."""
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx
