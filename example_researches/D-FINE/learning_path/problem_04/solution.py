import torch
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).mean()


def varifocal_loss(pred_logit, gt_score, label, num_classes, alpha=0.75, gamma=2.0):
    pred_score = torch.sigmoid(pred_logit)                          # (N, C)
    one_hot = F.one_hot(label, num_classes=num_classes).float()     # (N, C)

    # Soft target: positive = IoU score, negative = 0
    target = one_hot * gt_score.unsqueeze(-1)                       # (N, C)

    # Weight: positive = gt_score, negative = alpha * p^gamma
    weight = alpha * pred_score.pow(gamma) * (1 - one_hot) + one_hot * gt_score.unsqueeze(-1)

    return F.binary_cross_entropy_with_logits(
        pred_logit, target, weight=weight.detach(), reduction="none"
    )
