import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class BipartiteCostMatrix(nn.Module):
    def __init__(self, weight_dict={"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}):
        super().__init__()
        self.weights = weight_dict

    def forward(self, pred_logits, pred_boxes, gt_labels, gt_boxes):
        # Flatten batch
        bs, num_queries = pred_logits.shape[:2]
        
        out_prob = pred_logits.flatten(0, 1).sigmoid()
        out_bbox = pred_boxes.flatten(0, 1)

        # Class cost (focal)
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]

        # Bbox cost (L1)
        cost_bbox = torch.cdist(out_bbox, gt_boxes, p=1)

        C = self.weights["loss_ce"] * cost_class + self.weights["loss_bbox"] * cost_bbox
        C = C.view(bs, num_queries, -1)
        return C
