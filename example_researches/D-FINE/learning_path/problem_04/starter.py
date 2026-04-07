import torch
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """
    Sigmoid Focal Loss: addresses class imbalance by re-weighting easy vs hard examples.
    
    Args:
        inputs: Raw logits of shape (..., ) or shape matching targets
        targets: Binary targets of shape matching inputs
        alpha: Weight for positive class (default: 0.25)
        gamma: Exponent for focusing parameter (default: 2.0)
    
    Returns:
        Scalar loss value
    """
    raise NotImplementedError


def varifocal_loss(pred_logit, gt_score, label, num_classes, alpha=0.75, gamma=2.0):
    """
    Varifocal Loss: Focal loss variant that uses IoU quality scores as soft targets.
    Useful for joint classification and localization in object detection.
    
    Args:
        pred_logit: Raw logits of shape (N, C)
        gt_score: IoU-based quality scores of shape (N,), values in [0, 1]
        label: Ground truth class indices of shape (N,), int64
        num_classes: Number of classes (C)
        alpha: Weight parameter (default: 0.75)
        gamma: Focusing parameter (default: 2.0)
    
    Returns:
        Element-wise loss of shape (N, C)
    """
    raise NotImplementedError
