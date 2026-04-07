import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DFINEMini(nn.Module):
    """
    Minimal D-FINE Model: Backbone → Encoder → Decoder → Heads
    Chains all previously implemented modules into end-to-end detection pipeline.
    
    Args:
        num_classes: Number of object classes (default: 80 for COCO)
        num_queries: Number of object query slots (default: 100)
        d_model: Embedding dimension (default: 256)
        num_encoder_layers: Number of transformer encoder layers (default: 2)
        num_decoder_layers: Number of transformer decoder layers (default: 3)
    """
    
    def __init__(self, num_classes=80, num_queries=100, d_model=256,
                 num_encoder_layers=2, num_decoder_layers=3):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes

    def forward(self, images):
        """
        Args:
            images: Batch of images of shape (B, 3, H, W), H=W=256 for standard testing
        
        Returns:
            dict with:
                'pred_logits': (B, num_queries, num_classes) class logits
                'pred_boxes': (B, num_queries, 4) predicted boxes in normalized cxcywh, values in [0, 1]
        """
        raise NotImplementedError
