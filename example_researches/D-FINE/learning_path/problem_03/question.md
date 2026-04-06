# Problem 03 Questions

## Multiple Choice

1. What causes vanishing gradients for foreground boxes in dense detectors if cross-entropy is used?
A. Overwhelming count of easy background boxes dominating the loss sum.
B. The use of Sigmoid instead of Softmax.
C. The image resolution is too low.
D. The presence of non-linear activations.

2. In the focal loss paper, what does the $\gamma$ (gamma) parameter control?
A. Learning rate of the optimizer.
B. How steeply the loss decays for confident (easy) predictions.
C. The balancing ratio between positive and negative labels.
D. The width of the bounding box.

3. Standard cross-entropy uses the numerically stable `log-sum-exp` trick under the hood in PyTorch via `BCEWithLogitsLoss`. Why do we compute base `ce_loss` first before applying focal modulators?
A. To avoid computing sigmoid directly which can cause overflow on large logits.
B. Because it reduces parameter counts.
C. It allows caching the loss for backpropagation explicitly.
D. PyTorch enforces it.

4. If an example is classified perfectly ($p_t=1.0$) and $\gamma=2.0$, what is the modulating factor $(1 - p_t)^\gamma$?
A. 1.0
B. 0.5
C. 0.0
D. 2.0

5. What does the `alpha` ($\alpha$) parameter do?
A. It punishes bounding boxes whose size is too small.
B. It provides a constant class-imbalance weight to separate positive and negative labels independent of prediction confidence.
C. It is the bounding box regression scale.
D. It normalizes image color channels.

## Answer Key
1.A 2.B 3.A 4.C 5.B