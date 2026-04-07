# Problem 02 Questions

## Multiple Choice

1. What is the range of GIoU values?
   A. `[0, 1]`
   B. `[-1, 0]`
   C. `[-1, 1]`
   D. `(-∞, 1]`

2. When two predicted boxes perfectly overlap their ground-truth, what value does GIoU return?
   A. `0`
   B. `0.5`
   C. `1`
   D. `-1`

3. Why is the enclosing box needed in the GIoU formula?
   A. To normalize the prediction heads
   B. To provide gradient signal when boxes do not overlap (IoU=0)
   C. To increase bounding box regression speed
   D. To replace L1 loss in the model

4. In D-FINE's Hungarian Matcher, the GIoU cost is used as:
   A. `cost_giou = GIoU`
   B. `cost_giou = 1 - GIoU`
   C. `cost_giou = -GIoU`
   D. `cost_giou = exp(-GIoU)`

5. If `boxes1` has shape `(N, 4)` and `boxes2` has shape `(M, 4)`, what is the shape of the pairwise GIoU matrix?
   A. `(N+M, 4)`
   B. `(N,)`
   C. `(N, M)`
   D. `(M, N, 4)`

## Answer Key
1.C 2.C 3.B 4.C 5.C
