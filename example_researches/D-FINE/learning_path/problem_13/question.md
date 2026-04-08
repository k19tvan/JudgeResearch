## Question 13: Matching Union Implementation

Implement the `compute_matching_union()` method in the `HungarianMatcher` class to create consensus matching across multiple layers.

### Background
D-FINE uses outputs from multiple decoder and encoder layers for multi-layer supervision. Each layer computes its own Hungarian matching. However, different layers may produce different optimal matchings due to varying feature quality.

The "Global Oracle" (GO) combines these matchings by finding consensus pairs—matches that appear in multiple layers are more reliable for computing box regression losses (L1, GIoU).

### Specifications

```python
def compute_matching_union(self, indices_list):
    """Compute matching consensus for multi-layer supervision.
    
    Args:
        indices_list: List of matching results from all layers.
            Each element is a tuple (src_indices, tgt_indices):
            - src_indices: (num_matched,) tensor of prediction query indices
            - tgt_indices: (num_matched,) tensor of target object indices
            Length of list = number of layers (typically 4-5)
    
    Returns:
        matching_union: List of matching tuples for consensus
            Structure: [(src_batch_0, tgt_batch_0), (src_batch_1, tgt_batch_1), ...]
            One tuple per batch item showing consensus matches across all layers
    """
```

### Implementation Steps

1. **Process each batch item**
   - Matching indices are batch-wise: indices_list[layer_idx] has batch dimension
   - Iterate through each batch to build consensus separately

2. **Vote across layers** for each batch
   - Use a dictionary: `{(src, tgt): count}` to track frequency
   - For each layer's matching, increment count for each (src, tgt) pair

3. **Filter for consensus**
   - Keep pairs matching in >= 2 layers (adjustable threshold)
   - If no pairs meet threshold, fall back to any match

4. **Enforce one-to-one constraint**
   - Each query (src) can match at most one target
   - Each target (tgt) can match at most one query
   - Sort by frequency (prefer matches in more layers)
   - Greedily assign matches in frequency order

5. **Create output tensors**
   - Convert final lists to PyTorch tensors
   - Maintain consistent data types and devices

### Hints

- After finding consensus matches, you need to resolve conflicts where one query matches multiple targets
- When there's perfect disagreement (no consensus), return single-layer matches to avoid losing supervision
- Remember to move results back to the original device (not CPU)
- The output format should exactly match input format: List of (src_tensor, tgt_tensor) tuples

### Expected Behavior

**Input Example** (3 layers, batch_size=2):
```
Layer 0: [([2, 5, 7], [0, 1, 2]), ([1, 3], [0, 1])]  # batch 0: 3 matches, batch 1: 2 matches  
Layer 1: [([2, 5, 9], [0, 1, 3]), ([3, 5], [0, 2])]  # different matches
Layer 2: [([2, 5, 7], [0, 1, 2]), ([1, 3], [1, 0])]  # batch 0 same as layer 0
```

**Expected Output** (consensus):
```
Batch 0: ([2, 5, 7], [0, 1, 2])   # All 3 pairs appear in layers 0 and 2
Batch 1: ([1, 3, 5], [0, 1, 2])   # Resolved from conflicting layers
```

