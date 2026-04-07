# Problem 05 Questions

## Multiple Choice

1. Why are only matched predictions supervised with box regression (L1 and GIoU) losses, while ALL predictions receive VFL classification loss?
   A. Box loss is more expensive to compute
   B. Unmatched predictions should predict background class, not regress boxes — supervising their boxes would conflict with random initialization
   C. GIoU requires cxcywh format that only matched queries have
   D. L1 loss is undefined for unmatched predictions

2. In D-FINE's full criterion, the same `SetCriterion.forward` is called multiple times. Why?
   A. To compute different losses each time
   B. To apply multi-layer auxiliary supervision to intermediate decoder outputs and encoder proposals
   C. To average over different batch sizes
   D. Because PyTorch requires multiple forward passes for gradient accumulation

3. What does `_get_src_permutation_idx` produce?
   A. A sorted list of prediction indices
   B. `(batch_idx, src_idx)` — global flat indices for advanced tensor indexing across the batch
   C. The indices of unmatched background queries
   D. The Hungarian assignment cost values

4. The `num_boxes` normalization factor in D-FINE is computed as:
   A. `B * N` (batch × queries)
   B. `sum of T_i` across all images in the batch (total GT count)
   C. `max(T_i)` across the batch
   D. Fixed constant of 100

5. Why does D-FINE compute VFL loss over ALL N queries (not just matched ones)?
   A. To ensure unmatched queries learn to predict background confidence
   B. To speed up loss computation by avoiding index selection
   C. Because Hungarian Matcher returns N indices
   D. Because VFL only supports full-tensor inputs

## Answer Key
1.B 2.B 3.B 4.B 5.A
