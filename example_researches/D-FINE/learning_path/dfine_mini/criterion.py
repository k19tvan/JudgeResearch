"""Problem 05: D-FINE Set Criterion for Multi-Layer Loss Computation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from box_ops import box_cxcywh_to_xyxy
from iou import generalized_box_iou, box_iou
from losses import varifocal_loss
from fgl_loss import unimodal_distribution_focal_loss, bbox2distance


class SetCriterion(nn.Module):
    """Computes losses for Multi-Layer D-FINE.
    
    Supports:
    - Single-layer mode (for compatibility)
    - Multi-layer mode with auxiliary supervision
    - Matching union (consensus matching across layers)
    - Fine-Grained Localization (FGL) loss
    
    Key differences from simplified approach:
    - Gets indices from matcher (internally or externally)
    - Handles aux_outputs and enc_aux_outputs dicts
    - Computes matching union for box/FGL losses
    - Weights auxiliary layer losses lower (0.5x)
    """
    
    def __init__(
        self, 
        num_classes, 
        matcher=None,
        weight_vfl=1.0, 
        weight_bbox=5.0, 
        weight_giou=2.0,
        weight_fgl=0.0,
        reg_max=32,
        auxiliary_weight=0.5,
    ):
        """
        Args:
            num_classes: Number of object classes
            matcher: HungarianMatcher instance (optional, can be passed at forward)
            weight_vfl: Weight for VFL loss
            weight_bbox: Weight for L1 box loss
            weight_giou: Weight for GIoU loss
            weight_fgl: Weight for FGL loss (0 means skip FGL)
            reg_max: Fine-grained localization bins (reg_max+1 total bins)
            auxiliary_weight: Loss scale for auxiliary layers (0.5 = 50% of main layer)
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_vfl = weight_vfl
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_fgl = weight_fgl
        self.reg_max = reg_max
        self.auxiliary_weight = auxiliary_weight

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _compute_losses_single(self, outputs, targets, indices, num_boxes, layer_name=""):
        """Compute VFL, bbox, giou losses for a single layer."""
        losses = {}
        
        pred_logits = outputs["pred_logits"]  # (B, N, C)
        pred_boxes = outputs["pred_boxes"]    # (B, N, 4)
        B, N, C = pred_logits.shape
        device = pred_logits.device

        batch_idx, src_idx = self._get_src_permutation_idx(indices)

        # Gather matched predictions and targets
        matched_src = pred_boxes[batch_idx, src_idx] if src_idx.numel() > 0 else torch.zeros(
            (0, 4), device=device
        )
        matched_tgt = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0) \
            if src_idx.numel() > 0 else torch.zeros((0, 4), device=device)

        # L1 loss
        if src_idx.numel() > 0:
            loss_bbox = F.l1_loss(matched_src, matched_tgt, reduction="sum") / num_boxes
        else:
            loss_bbox = torch.tensor(0.0, device=device)

        # GIoU loss
        if src_idx.numel() > 0:
            src_xyxy = box_cxcywh_to_xyxy(matched_src)
            tgt_xyxy = box_cxcywh_to_xyxy(matched_tgt)
            giou = torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))
            loss_giou = (1 - giou).sum() / num_boxes
        else:
            loss_giou = torch.tensor(0.0, device=device)

        # VFL loss
        target_classes = torch.full((B, N), self.num_classes, dtype=torch.int64, device=device)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes[batch_idx, src_idx] = target_classes_o

        # IoU-based quality scores for VFL
        gt_score = torch.zeros(B, N, device=device)
        if src_idx.numel() > 0:
            ious = torch.diag(generalized_box_iou(src_xyxy.detach(), tgt_xyxy))
            gt_score[batch_idx, src_idx] = ious.clamp(0).detach()

        # Compute VFL loss
        label_clip = target_classes.flatten().clamp(0, C - 1)
        vfl = varifocal_loss(
            pred_logits.flatten(0, 1),
            gt_score.flatten(),
            label_clip,
            num_classes=C
        )
        
        # Normalize: per-query mean, then aggregate by num_boxes
        vfl = vfl.mean(dim=-1) if vfl.dim() > 1 else vfl  # Per-query normalization
        
        # Zero out background gradients
        bg_mask = (target_classes.flatten() == self.num_classes)
        vfl[bg_mask] = vfl[bg_mask].detach()
        loss_vfl = vfl.sum() / num_boxes

        # Store losses with layer name suffix if provided
        suffix = f"_{layer_name}" if layer_name else ""
        losses[f"loss_vfl{suffix}"] = self.weight_vfl * loss_vfl
        losses[f"loss_bbox{suffix}"] = self.weight_bbox * loss_bbox
        losses[f"loss_giou{suffix}"] = self.weight_giou * loss_giou

        return losses, (src_xyxy.detach() if src_idx.numel() > 0 else None), ious.detach() \
            if src_idx.numel() > 0 else None

    def _compute_fgl_loss(self, outputs, targets, indices_go, num_boxes):
        """Compute Fine-Grained Localization loss using matching union indices."""
        if "pred_corners" not in outputs or self.weight_fgl == 0:
            return {}

        pred_corners = outputs["pred_corners"]  # (B, N, 4, reg_max+1)
        ref_points = outputs.get("ref_points")  # (B, N, 2)
        
        if ref_points is None or pred_corners is None:
            return {}

        batch_idx, src_idx = self._get_src_permutation_idx(indices_go)
        
        if src_idx.numel() == 0:
            return {}

        # Get matched boxes and compute distance distributions
        matched_tgt = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices_go)], dim=0)
        ref_pts_matched = ref_points[batch_idx, src_idx]  # (T, 2)
        
        # Convert boxes to distance distributions
        distances, soft_label, dist_weight = bbox2distance(
            ref_pts_matched,
            matched_tgt,
            reg_max=self.reg_max
        )

        # Extract matched prediction corners
        pred_corners_matched = pred_corners[batch_idx, src_idx]  # (T, 4, reg_max+1)
        
        # Compute FGL loss
        loss_fgl = unimodal_distribution_focal_loss(
            pred_corners_matched,
            soft_label,
            weight=dist_weight,
            reduction='mean'
        )

        return {"loss_fgl": self.weight_fgl * loss_fgl}

    def forward(self, outputs, targets, indices=None, matcher=None):
        """
        Flexible forward pass supporting both modes:
        
        Mode 1 (Simple, single-layer):
            criterion(outputs, targets, indices=indices_from_matcher)
        
        Mode 2 (Full, multi-layer):
            criterion(outputs, targets, matcher=matcher)
            -> internally calls matcher for each layer + computes union
        
        Args:
            outputs: Dict with pred_logits, pred_boxes, and optionally aux_outputs, etc.
            targets: List of target dicts
            indices: Pre-computed matching indices (mode 1) or None (mode 2)
            matcher: Matcher instance to use (mode 2)
        
        Returns:
            losses: Dict of aggregated losses
        """
        matcher = matcher or self.matcher
        
        # Determine mode
        if indices is not None:
            # Mode 1: Single-layer, indices provided
            num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
            losses, _, _ = self._compute_losses_single(outputs, targets, indices, num_boxes)
            
            # Add FGL loss if present
            if "pred_corners" in outputs:
                losses.update(self._compute_fgl_loss(outputs, targets, indices, num_boxes))
            
            return losses
        
        # Mode 2: Multi-layer with matcher
        if matcher is None:
            raise ValueError(
                "Either provide indices or matcher for multi-layer loss computation"
            )
        
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        all_losses = {}

        # Get matching for final layer
        indices_final = matcher(outputs, targets)["indices"]
        
        # Compute losses for final layer
        losses_final, _, _ = self._compute_losses_single(
            outputs, targets, indices_final, num_boxes
        )
        all_losses.update(losses_final)

        # Collect indices from all layers for matching union
        all_indices_list = [indices_final]

        # Process auxiliary outputs from decoder
        if "aux_outputs" in outputs:
            for layer_idx, aux_out in enumerate(outputs["aux_outputs"]):
                indices_aux = matcher(aux_out, targets)["indices"]
                all_indices_list.append(indices_aux)
                
                # Compute losses for this auxiliary layer
                losses_aux, _, _ = self._compute_losses_single(
                    aux_out, targets, indices_aux, num_boxes, layer_name=f"aux_{layer_idx}"
                )
                # Weight auxiliary losses
                losses_aux = {k: v * self.auxiliary_weight for k, v in losses_aux.items()}
                all_losses.update(losses_aux)

        # Process auxiliary outputs from encoder
        if "enc_aux_outputs" in outputs:
            for layer_idx, enc_out in enumerate(outputs["enc_aux_outputs"]):
                indices_enc = matcher(enc_out, targets)["indices"]
                all_indices_list.append(indices_enc)
                
                # Compute losses for this encoder layer
                losses_enc, _, _ = self._compute_losses_single(
                    enc_out, targets, indices_enc, num_boxes, layer_name=f"enc_aux_{layer_idx}"
                )
                # Weight auxiliary losses lower
                losses_enc = {k: v * self.auxiliary_weight for k, v in losses_enc.items()}
                all_losses.update(losses_enc)

        # Compute matching union (GO indices) for box/FGL losses
        if len(all_indices_list) > 1 and hasattr(matcher, 'compute_matching_union'):
            indices_go = matcher.compute_matching_union(all_indices_list)
        else:
            indices_go = indices_final

        # Compute FGL loss on matching union
        if "pred_corners" in outputs:
            losses_fgl = self._compute_fgl_loss(outputs, targets, indices_go, num_boxes)
            all_losses.update(losses_fgl)

        return all_losses
