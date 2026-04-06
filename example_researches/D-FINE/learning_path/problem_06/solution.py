import torch
import torch.nn as nn

class DenoisingGenerator(nn.Module):
    """
    Problem 06: Contrastive Denoising (CDN)
    """
    def __init__(self, hidden_dim=256, num_classes=80):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes + 1, hidden_dim)

    def forward(self, gt_boxes, gt_labels, noise_scale=0.1):
        """
        gt_boxes: [N, 4] in cxcywh
        gt_labels: [N]
        Returns noisy queries
        """
        noisy_boxes = gt_boxes.clone() + (torch.rand_like(gt_boxes) - 0.5) * noise_scale
        noisy_labels = gt_labels.clone()
        
        # Simple flip noise for labels (contrastive part)
        flip_mask = torch.rand_like(noisy_labels, dtype=torch.float32) < 0.2
        noisy_labels[flip_mask] = torch.randint_like(noisy_labels[flip_mask], 0, 80)
        
        box_emb = noisy_boxes # In real impl, apply sine embedding
        label_emb = self.label_emb(noisy_labels)
        
        return noisy_boxes, label_emb
