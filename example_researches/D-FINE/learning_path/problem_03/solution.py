import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5*w, y_c - 0.5*h, x_c + 0.5*w, y_c + 0.5*h], dim=-1)


def _box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    a1 = _box_area(boxes1)
    a2 = _box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = a1[:, None] + a2 - inter
    iou = inter / union.clamp(min=1e-6)
    elt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    erb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    ewh = (erb - elt).clamp(min=0)
    encl = ewh[:, :, 0] * ewh[:, :, 1]
    return iou - (encl - union) / encl.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions
        out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))   # (B*N, C)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)                # (B*N, 4)

        tgt_ids  = torch.cat([v["labels"] for v in targets])          # (T_total,)
        tgt_bbox = torch.cat([v["boxes"]  for v in targets])          # (T_total, 4)

        # Focal classification cost
        out_prob_sel = out_prob[:, tgt_ids]                            # (B*N, T_total)
        neg_cost = (1-self.alpha) * (out_prob_sel**self.gamma) * (-(1-out_prob_sel+1e-8).log())
        pos_cost = self.alpha * ((1-out_prob_sel)**self.gamma) * (-(out_prob_sel+1e-8).log())
        cost_class = pos_cost - neg_cost

        # L1 box cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Combined cost
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = torch.nan_to_num(C, nan=1.0)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]
        return {"indices": indices}
