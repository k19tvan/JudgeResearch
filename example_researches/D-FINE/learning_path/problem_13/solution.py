"""Problem 13: Solution - Matching Union for Multi-Layer Supervision"""
import torch
import torch.nn as nn
from typing import List, Tuple


class HungarianMatcher(nn.Module):
    """Hungarian Matcher with matching union computation."""
    
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        """Hungarian matching - implemented in Problem 03."""
        raise NotImplementedError("Implement single-layer matching in Problem 03")
    
    def compute_matching_union(self, indices_list: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Compute matching union (consensus across layers).
        
        Creates Global Oracle (GO) indices by finding consensus matches across
        multiple decoder/encoder layers.
        
        Args:
            indices_list: List of (src_indices, tgt_indices) from each layer
                         Each element: (src tensor, tgt tensor) for one layer
                         List length = number of layers (e.g., 4-5 for D-FINE)
        
        Returns:
            matching_union: List of (src_consensus, tgt_consensus) tuples
                           One tuple per batch item with consensus matches
        """
        if not indices_list:
            return []
        
        batch_size = len(indices_list[0])
        matching_union = []
        
        # Process each batch item separately
        for batch_idx in range(batch_size):
            # Step 1: Collect all matches from all layers for this batch
            all_matches = {}  # Key: (src, tgt), Value: count (how many layers matched this pair)
            
            for layer_indices in indices_list:
                src_idx, tgt_idx = layer_indices
                # Iterate through all matches in this layer
                for src, tgt in zip(src_idx, tgt_idx):
                    src_val = src.item()
                    tgt_val = tgt.item()
                    key = (src_val, tgt_val)
                    all_matches[key] = all_matches.get(key, 0) + 1
            
            # Step 2: Filter for consensus (matches in >= 2 layers)
            consensus_threshold = 2
            consensus_matches = {k: v for k, v in all_matches.items() if v >= consensus_threshold}
            
            # Step 3: If no consensus, use all matches (don't lose supervision)
            if not consensus_matches:
                consensus_matches = all_matches
            
            # Step 4: Enforce one-to-one constraint
            # Each src can match at most one target, each target at most one src
            final_src = []
            final_tgt = []
            used_queries = set()
            used_targets = set()
            
            # Sort by frequency (descending) then by keys for determinism
            sorted_matches = sorted(
                consensus_matches.items(),
                key=lambda x: (-x[1], x[0][0], x[0][1])
            )
            
            # Greedily assign matches in order of frequency
            for (src, tgt), freq in sorted_matches:
                if src not in used_queries and tgt not in used_targets:
                    final_src.append(src)
                    final_tgt.append(tgt)
                    used_queries.add(src)
                    used_targets.add(tgt)
            
            # Step 5: Convert to tensors with correct device and dtype
            if final_src:
                device = indices_list[0][0].device
                final_src_tensor = torch.tensor(final_src, dtype=torch.int64, device=device)
                final_tgt_tensor = torch.tensor(final_tgt, dtype=torch.int64, device=device)
            else:
                # Empty matching for this batch item
                device = indices_list[0][0].device
                final_src_tensor = torch.tensor([], dtype=torch.int64, device=device)
                final_tgt_tensor = torch.tensor([], dtype=torch.int64, device=device)
            
            matching_union.append((final_src_tensor, final_tgt_tensor))
        
        return matching_union
