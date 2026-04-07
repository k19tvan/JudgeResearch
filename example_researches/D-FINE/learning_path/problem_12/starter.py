import torch
import torch.nn as nn
from typing import List


def make_synthetic_batch(B: int, num_gt: int, num_classes: int = 10,
                          device: torch.device = torch.device("cpu")):
    """
    Generates a synthetic batch of images and COCO-format targets.

    Args:
        B:           Batch size.
        num_gt:      Number of ground-truth objects per image.
        num_classes: Total class count.
        device:      torch.device.

    Returns:
        images:  (B, 3, 256, 256) float tensor.
        targets: list of B dicts with 'labels' (T,) and 'boxes' (T, 4) cxcywh.
    """
    raise NotImplementedError


def train_one_epoch(model, criterion, matcher, optimizer, data_loader, device):
    """
    Runs one full training epoch.

    Args:
        model:       DFINEMini in train mode.
        criterion:   SetCriterion.
        matcher:     HungarianMatcher.
        optimizer:   torch.optim optimizer.
        data_loader: Iterable of (images, targets) batches.
        device:      torch.device.

    Returns:
        dict with 'avg_loss_vfl', 'avg_loss_bbox', 'avg_loss_giou'.
    """
    raise NotImplementedError


def evaluate(model, data_loader, device):
    """
    Evaluates model with mean IoU metric.

    Args:
        model:       DFINEMini in eval mode.
        data_loader: Iterable of (images, targets) batches.
        device:      torch.device.

    Returns:
        dict with 'mean_iou' (float).
    """
    raise NotImplementedError
