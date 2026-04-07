"""Problem 05: D-FINE Set Criterion for Loss Computation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from box_ops import box_cxcywh_to_xyxy
from iou import generalized_box_iou, box_iou
from losses import varifocal_loss


class SetCriterion(nn.Module):
    """Computes losses for D-FINE: VFL, L1 box, and GIoU losses."""
    
    def __init__(self, num_classes, weight_vfl=1.0, weight_bbox=5.0, weight_giou=2.0):
        super().__init__()
        self.num_classes  = num_classes
        self.weight_vfl   = weight_vfl
        self.weight_bbox  = weight_bbox
        self.weight_giou  = weight_giou

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i,(src,_) in enumerate(indices)])
        src_idx   = torch.cat([src for (src,_) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"]  # (B, N, C)
        pred_boxes  = outputs["pred_boxes"]   # (B, N, 4)
        B, N, C     = pred_logits.shape
        device      = pred_logits.device

        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        batch_idx, src_idx = self._get_src_permutation_idx(indices)

        # Gather matched boxes
        matched_src = pred_boxes[batch_idx, src_idx]                                    # (T,4)
        matched_tgt = torch.cat([t["boxes"][j] for t,(_,j) in zip(targets, indices)])  # (T,4)

        # L1 loss
        loss_bbox = F.l1_loss(matched_src, matched_tgt, reduction="sum") / num_boxes

        # GIoU loss
        src_xyxy = box_cxcywh_to_xyxy(matched_src)
        tgt_xyxy = box_cxcywh_to_xyxy(matched_tgt)
        giou = torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))
        loss_giou = (1 - giou).sum() / num_boxes

        # VFL loss — over all N queries
        target_classes = torch.full((B, N), self.num_classes, dtype=torch.int64, device=device)
        target_classes_o = torch.cat([t["labels"][j] for t,(_,j) in zip(targets, indices)])
        target_classes[batch_idx, src_idx] = target_classes_o

        # IoU-based quality scores
        gt_score = torch.zeros(B, N, device=device)
        if matched_src.numel() > 0:
            ious = torch.diag(generalized_box_iou(src_xyxy.detach(), tgt_xyxy))
            gt_score[batch_idx, src_idx] = ious.clamp(0).detach()

        # Clip target_classes to [0, C-1] for one_hot — treat background as 0 placeholder
        label_clip = target_classes.flatten().clamp(0, C-1)
        vfl = varifocal_loss(
            pred_logits.flatten(0,1),
            gt_score.flatten(),
            label_clip,
            num_classes=C
        )
        # Zero out background columns
        bg_mask = (target_classes.flatten() == self.num_classes)
        vfl[bg_mask] = vfl[bg_mask].detach()   # no class gradient for background
        loss_vfl = vfl.sum() / num_boxes

        return {
            "loss_vfl":  self.weight_vfl  * loss_vfl,
            "loss_bbox": self.weight_bbox * loss_bbox,
            "loss_giou": self.weight_giou * loss_giou,
        }
