# Problem 03 - Focal Loss

## Description
- In object detection, most of the image is background. Standard Cross Entropy Loss would be overwhelmed by easy, negative background examples.
- Focal loss addresses this extreme class imbalance by dynamically scaling down the loss based on prediction confidence, focusing the model on hard, misclassified examples.
- Your task is to implement the binary signature `sigmoid_focal_loss` which D-FINE uses in its Hungarian matching.

### Data Specification and Shapes
- `inputs`: Tensor of shape `(N, C)` where N is samples, C is number of classes. Unnormalized logits.
- `targets`: Tensor of shape `(N, C)` containing binary labels (one-hot encoded).
- `alpha`, `gamma`: Scaling factors.

## Requirements
- Implement `sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor`
- Do not use `for` loops. Must be fully vectorized.

## Hints
- `prob = inputs.sigmoid()`
- The cross-entropy term is `ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")`.
- The modulation factor is `p_t = prob * targets + (1 - prob) * (1 - targets)`.
- The loss equation applies `(1 - p_t) ** gamma` to `ce_loss`.
- Don't forget the alpha weighting: `alpha_t = alpha * targets + (1 - alpha) * (1 - targets)`.

## Checker
Run the provided checker to validate your implementation:
`python checker.py`