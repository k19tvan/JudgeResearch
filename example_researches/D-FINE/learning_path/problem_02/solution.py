import torch
from torch import Tensor


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    # Broadcast: (N, 1, 2) vs (M, 2) → (N, M, 2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)          # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]    # (N, M)

    union = area1[:, None] + area2 - inter  # (N, M)
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 has degenerate boxes"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 has degenerate boxes"

    iou, union = box_iou(boxes1, boxes2)

    # Enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    encl_area = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    return iou - (encl_area - union) / encl_area.clamp(min=1e-6)
