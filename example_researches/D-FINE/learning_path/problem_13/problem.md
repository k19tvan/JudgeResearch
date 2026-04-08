# Problem 13: Matching Union (Global Oracle Consensus)

## Objective
Implement **matching union** logic to create consensus matching across multiple decoder/encoder layers. D-FINE uses outputs from ALL layers for multi-layer supervision, but each layer produces different optimal matchings. This problem focuses on finding consensus matches across layers.

## Why Matching Union?

In D-FINE's multi-layer training:
- **Encoder Layer 1** matches predictions to targets → indices_enc1
- **Encoder Layer 2** matches predictions to targets → indices_enc2  
- **Decoder Layer 1** matches predictions to targets → indices_dec1
- **Decoder Layer 2** matches predictions to targets → indices_dec2
- **Decoder Layer 3 (final)** matches predictions to targets → indices_final

Different layers see different feature quality, so they produce different matchings!

**The Problem**: Which indices should we use for box losses (L1, GIoU)?
- **Answer**: Use indices where multiple layers agree (matching UNION / Global Oracle)

**Key Insight**: If a specific prediction-target pair is matched in 2+ layers, it's a reliable assignment → use it for box losses. If matched in only 1 layer, use single-layer indices only.

## Concepts

### Matching Consensus
```python
# Layer indices:
indices_layer_1 = [(pred_idx=[2,5,7], tgt_idx=[0,1,2]), ...]
indices_layer_2 = [(pred_idx=[3,5,9], tgt_idx=[0,1,3]), ...]
indices_layer_3 = [(pred_idx=[2,5,7], tgt_idx=[0,1,2]), ...]

# Consensus voting:
# Pair (pred=2, tgt=0) appears in layers 1 and 3 → CONSENSUS → keep
# Pair (pred=5, tgt=1) appears in layers 1, 2, 3 → STRONG CONSENSUS → keep
# Pair (pred=3, tgt=0) appears only in layer 2 → WEAK → maybe skip
```

### One-to-One Matching Constraint
After finding consensus, enforce one-to-one:
- Each `pred_idx` can match at most 1 target
- Each `tgt_idx` can match at most 1 prediction
- Resolve conflicts by frequency in consensus (highest priority first)

## Your Task

Implement `compute_matching_union()` method in `HungarianMatcher`:

```python
def compute_matching_union(self, indices_list):
    """
    Args:
        indices_list: List of (src_indices, tgt_indices) tuples
                      - src_indices: (num_matched,) query indices
                      - tgt_indices: (num_matched,) target indices
                      Length = num_layers (typically 4-5 for D-FINE)
    
    Returns:
        matching_union: List of (final_src, final_tgt) with consensus matches
    """
```

### Algorithm
1. **Collect all matches across layers**
   ```python
   all_matches = {}  # (src, tgt) -> count
   for each layer:
       for each (src, tgt) pair in layer_indices:
           all_matches[(src, tgt)] += 1
   ```

2. **Filter for consensus** (appears in 2+ layers)
   ```python
   consensus_matches = {k: v for k, v in all_matches.items() if v >= 2}
   ```

3. **Handle empty consensus** (all layers disagree)
   - If no consensus found, keep all matches from all_matches
   - This ensures we never return empty matching

4. **Enforce one-to-one**
   ```python
   final_src = []
   final_tgt = []
   used_queries = set()
   used_targets = set()
   
   # Sort by frequency descending (most agreed-upon first)
   sorted_matches = sorted(consensus_matches.items(), 
                          key=lambda x: -x[1])
   
   for (src, tgt), freq in sorted_matches:
       if src not in used_queries and tgt not in used_targets:
           final_src.append(src)
           final_tgt.append(tgt)
           used_queries.add(src)
           used_targets.add(tgt)
   ```

5. **Return as tensor tuples**
   ```python
   return [(torch.tensor(src), torch.tensor(tgt)), ...]
   ```

## Integration

The criterion uses this in Mode 2 (multi-layer):
```python
# In SetCriterion.forward():
all_indices_list = [indices_final, indices_aux_1, indices_aux_2, ...]
indices_go = matcher.compute_matching_union(all_indices_list)
# Use indices_go for box/FGL losses (more stable)
# Use single indices_final for classification (layer-specific)
```

## Test Cases

Your implementation must handle:
1. **Normal case**: All layers produce reasonable matchings
   - Input: 4 layers with 15-20 matches each
   - Output: 10-15 consensus matches
   
2. **Empty layers**: Some layers match no objects
   - Input: 4 layers, 2 with matches, 2 empty
   - Output: Still produces valid matching from non-empty layers
   
3. **Complete disagreement**: All layers match differently
   - Input: 4 layers with completely different matchings
   - Output: Falls back to keeping all matches, enforces one-to-one
   
4. **One-to-one enforcement**: Multiple matches to same query/target
   - Input: Overlapping matches after consensus
   - Output: Resolves conflicts, each query/target used once

## Resources
- See Problem 03 (`HungarianMatcher.forward()`) for single-layer matching
- Reference: `src/zoo/dfine/dfine_criterion.py` lines 245-268 for `_get_go_indices()`
