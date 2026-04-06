import torch
import torch.nn as nn

class PostProcessor(nn.Module):
    def __init__(self, topk=300):
        super().__init__()
        self.topk = topk

    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = out_logits.sigmoid()
        
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.topk, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
