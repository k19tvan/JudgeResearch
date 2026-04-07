"""Problem 11: End-to-End D-FINE Model Assembly"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from backbone import HGNetV2Stem
from positional_encoding import PositionEmbeddingSine2D
from encoder import TransformerEncoderLayer
from decoder import TransformerDecoderLayer


class DFINEMini(nn.Module):
    """Minimal D-FINE Model: Backbone → Encoder → Decoder → Heads."""
    
    def __init__(self, num_classes=80, num_queries=100, d_model=256,
                 num_encoder_layers=2, num_decoder_layers=3):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Backbone and projections
        self.backbone = HGNetV2Stem()
        self.proj     = nn.Conv2d(96, d_model, 1)
        self.pe       = PositionEmbeddingSine2D(d_model)
        
        # Encoder and decoder stacks
        self.enc = nn.ModuleList([
            TransformerEncoderLayer(d_model, 8, d_model*4) 
            for _ in range(num_encoder_layers)
        ])
        self.dec = nn.ModuleList([
            TransformerDecoderLayer(d_model, 8, d_model*4) 
            for _ in range(num_decoder_layers)
        ])
        
        # Learnable embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos   = nn.Embedding(num_queries, d_model)
        
        # Prediction heads
        self.cls_head  = nn.Linear(d_model, num_classes)
        self.box_head  = nn.Sequential(
            nn.Linear(d_model, d_model), 
            nn.ReLU(), 
            nn.Linear(d_model, 4)
        )

    def forward(self, images):
        """Forward pass through full pipeline."""
        B = images.shape[0]
        
        # Backbone
        _, feat = self.backbone(images)            # (B, 96, H', W')
        fp = self.proj(feat)                       # (B, d_model, H', W')
        
        # Flatten and get positional encodings
        src = fp.flatten(2).transpose(1, 2)        # (B, H'*W', d_model)
        pos = self.pe(fp)                          # (B, H'*W', d_model)
        
        # Encoder
        memory = src
        for enc in self.enc:
            memory = enc(memory, pos)
        
        # Decoder
        tgt  = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # (B, num_queries, d_model)
        qpos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)     # (B, num_queries, d_model)
        for dec in self.dec:
            tgt = dec(tgt, memory, qpos, pos)
        
        # Prediction heads
        pred_logits = self.cls_head(tgt)           # (B, num_queries, num_classes)
        pred_boxes  = self.box_head(tgt).sigmoid() # (B, num_queries, 4)
        
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
