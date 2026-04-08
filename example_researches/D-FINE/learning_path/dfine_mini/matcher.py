"""Problem 03: Hungarian Matcher for Bipartite Matching"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

from box_ops import box_cxcywh_to_xyxy
from iou import generalized_box_iou


class HungarianMatcher(nn.Module):
    """Performs optimal bipartite matching between predictions and ground-truth boxes."""
    
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions
        out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))   # (B*N, C)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)                # (B*N, 4)

        tgt_ids  = torch.cat([v["labels"] for v in targets])          # (T_total,)
        tgt_bbox = torch.cat([v["boxes"]  for v in targets])          # (T_total, 4)

        # Focal classification cost
        out_prob_sel = out_prob[:, tgt_ids]                            # (B*N, T_total)
        neg_cost = (1-self.alpha) * (out_prob_sel**self.gamma) * (-(1-out_prob_sel+1e-8).log())
        pos_cost = self.alpha * ((1-out_prob_sel)**self.gamma) * (-(out_prob_sel+1e-8).log())
        cost_class = pos_cost - neg_cost

        # L1 box cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Combined cost
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = torch.nan_to_num(C, nan=1.0)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]
        return {"indices": indices}
    
    def compute_matching_union(self, indices_list):
        """Compute matching union (Global Oracle) across multiple matching layers.
        
        The idea: Different decoder/encoder layers may produce different optimal matchings.
        This method finds consensus by keeping matches that appear consistently across layers.
        
        Algorithm:
        1. For each prediction query-target pair, count how many layers matched it
        2. Keep pairs matched in >= 2 layers (consensus)
        3. For pairs matched in only 1 layer, resolve conflicts by layer priority
        
        Args:
            indices_list: List of (src_indices, tgt_indices) tuples, one per layer
                         Each element is a tuple of two tensors of shape (num_matched,)
        
        Returns:
            matching_union: List of (src_indices, tgt_indices) tuples for consensus matching
        """
        batch_size = len(indices_list[0])
        matching_union = []
        
        # Process each batch item separately
        for batch_idx in range(batch_size):
            # Collect all matches from all layers for this batch
            all_matches = {}  # Key: (src, tgt), Value: count
            
            for layer_indices in indices_list:
                src_idx, tgt_idx = layer_indices
                for src, tgt in zip(src_idx, tgt_idx):
                    src_val = src.item()
                    tgt_val = tgt.item()
                    key = (src_val, tgt_val)
                    all_matches[key] = all_matches.get(key, 0) + 1
            
            # Keep only consensus matches (appearing in 2+ layers)
            # If a query appears in multiple matches at same layer, take with highest IoU
            consensus_matches = {k: v for k, v in all_matches.items() if v >= 2}
            
            # If no consensus, keep single-layer matches but prefer earlier layers (encoder)
            if not consensus_matches:
                consensus_matches = all_matches
            
            # Ensure one-to-one matching: each src matched to at most one target
            final_src = []
            final_tgt = []
            used_queries = set()
            used_targets = set()
            
            # Sort by frequency (highest first) then by key for determinism
            sorted_matches = sorted(
                consensus_matches.items(), 
                key=lambda x: (-x[1], x[0][0], x[0][1])
            )
            
            for (src, tgt), freq in sorted_matches:
                if src not in used_queries and tgt not in used_targets:
                    final_src.append(src)
                    final_tgt.append(tgt)
                    used_queries.add(src)
                    used_targets.add(tgt)
            
            # Convert to tensors
            if final_src:
                final_src = torch.tensor(final_src, dtype=torch.int64, device=indices_list[0][0].device)
                final_tgt = torch.tensor(final_tgt, dtype=torch.int64, device=indices_list[0][1].device)
            else:
                # Empty matching for this batch item
                device = indices_list[0][0].device
                final_src = torch.tensor([], dtype=torch.int64, device=device)
                final_tgt = torch.tensor([], dtype=torch.int64, device=device)
            
            matching_union.append((final_src, final_tgt))
        
        return matching_union
