# Problem 03 Questions

## Multiple Choice

1. Why does DETR use the Hungarian Algorithm instead of anchor-based matching?
   A. Anchors require more memory
   B. Hungarian produces a 1-to-1 unique assignment, eliminating duplicate prediction suppression (NMS)
   C. Hungarian is faster than anchor-based methods
   D. Hungarian works only for square images

2. What does `linear_sum_assignment` return?
   A. The minimum cost itself
   B. Row and column indices of the optimal assignment
   C. A boolean mask for selected predictions
   D. The cost matrix sorted by assignment

3. Why is the GIoU cost negated (`-GIoU`) in the matcher?
   A. Because `scipy` minimizes cost, and higher GIoU is better
   B. Because GIoU is always negative
   C. To match the sign convention of L1 loss
   D. Because predictions are in `cxcywh` format

4. In D-FINE's matcher, all cost computations are wrapped in `@torch.no_grad()`. Why?
   A. Costs are approximate and gradients would be wrong
   B. Matching does not need to be differentiable; it only selects indices
   C. `scipy` requires no-grad mode
   D. It prevents CUDA out-of-memory errors

5. If B=2, N=100, T_1=3, T_2=5, what is the shape of the full cost matrix `C` before splitting?
   A. `(200, 8)`
   B. `(2, 100, 8)`
   C. `(100, 8)`
   D. `(2, 8, 100)`

## Answer Key
1.B 2.B 3.A 4.B 5.B
