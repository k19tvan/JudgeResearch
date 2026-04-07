# Problem 04 Questions

## Multiple Choice

1. What key difference does VFL have over standard binary Focal Loss?
   A. VFL uses softmax instead of sigmoid
   B. VFL uses the predicted IoU as the positive class target score instead of 1
   C. VFL ignores background predictions
   D. VFL applies a stronger gamma for positive examples

2. In VFL, what is the weight applied to background (negative) predictions?
   A. Constant 1
   B. `alpha * (1 - p)^gamma`
   C. `alpha * p^gamma`
   D. `gt_score^gamma`

3. Why does VFL directly benefit detection quality (mAP) over standard Focal Loss?
   A. Because it reduces training time
   B. Because the confidence score is calibrated to localization quality (IoU), making ranking during evaluation more accurate
   C. Because it has fewer hyperparameters
   D. Because VFL avoids the `log` operation

4. The VFL target for a positive query matched to class `k` with IoU=0.6 is:
   A. `target = [0, ..., 1, ..., 0]` (one-hot at k)
   B. `target = [0, ..., 0.6, ..., 0]` (IoU score at k, 0 elsewhere)
   C. `target = 0.6` (scalar)
   D. `target = [0.6, ..., 0.6]` (uniform)

5. When implementing VFL, why is `F.binary_cross_entropy_with_logits` preferred over manually computing `log(sigmoid(x))`?
   A. It's faster on GPU
   B. It's numerically stable (avoids `log(0)` and overflow in `sigmoid` for extreme logits)
   C. It supports soft targets
   D. Both B and C

## Answer Key
1.B 2.C 3.B 4.B 5.D
