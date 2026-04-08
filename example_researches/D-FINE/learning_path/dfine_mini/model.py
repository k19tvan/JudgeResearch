"""Problem 12: End-to-End Multi-Layer D-FINE Model Assembly"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from backbone import HGNetV2Stem
from neck import SimpleNeck
from positional_encoding import PositionEmbeddingSine2D
from encoder import TransformerEncoderLayer
from decoder import TransformerDecoderLayer


class DFINEMini(nn.Module):
    """Multi-Layer D-FINE Model with Auxiliary Outputs and FGL Head.
    
    Architecture:
    - Backbone: Single-level feature extraction
    - Neck: Multi-level feature pyramid
    - Encoder: Stack of transformer layers (with auxiliary outputs)
    - Decoder: Stack of transformer layers (with auxiliary outputs)
    - Heads: Classification, bounding box, and FGL distribution heads
    
    Returns auxiliary outputs from each layer for multi-layer supervision.
    """
    
    def __init__(
        self, 
        num_classes=80, 
        num_queries=100, 
        d_model=256,
        num_encoder_layers=2, 
        num_decoder_layers=3,
        num_neck_levels=3,
        reg_max=32,
        backbone_out_channels=96,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.reg_max = reg_max
        
        # ===== Backbone =====
        self.backbone = HGNetV2Stem()  # Out: (B, 96, H', W')
        
        # ===== Neck (Multi-level Features) =====
        self.neck = SimpleNeck(
            c_in=backbone_out_channels,
            c_out=d_model,
            num_levels=num_neck_levels
        )
        # Neck outputs: List[(B, d_model, H_i, W_i)]
        # We'll process multi-level features in encoder
        
        # ===== Feature Flattening & Positional Encoding =====
        self.pe = PositionEmbeddingSine2D(d_model)
        
        # ===== Encoder Stack =====
        self.enc_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, 8, d_model * 4) 
            for _ in range(num_encoder_layers)
        ])
        
        # Auxiliary prediction heads for encoder outputs
        self.enc_aux_heads = nn.ModuleList([
            nn.ModuleDict({
                'cls': nn.Linear(d_model, num_classes),
                'box': nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 4)
                ),
            }) for _ in range(num_encoder_layers)
        ])
        
        # ===== Decoder Stack =====
        self.dec_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, 8, d_model * 4)
            for _ in range(num_decoder_layers)
        ])
        
        # Auxiliary prediction heads for decoder outputs
        self.dec_aux_heads = nn.ModuleList([
            nn.ModuleDict({
                'cls': nn.Linear(d_model, num_classes),
                'box': nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 4)
                ),
                'fgl': nn.Linear(d_model, 4 * (reg_max + 1)),  # FGL head
            }) for _ in range(num_decoder_layers)
        ])
        
        # ===== Main Prediction Heads (Final Layer) =====
        self.cls_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)
        )
        # FGL head: predict distribution over bins for each distance
        self.fgl_head = nn.Linear(d_model, 4 * (reg_max + 1))
        
        # ===== Learnable Embeddings =====
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)

    def forward(self, images, targets=None):
        """Forward pass with auxiliary outputs.
        
        Args:
            images: (B, 3, H, W) input images
            targets: (optional) for compatibility
        
        Returns:
            Dict with:
            - pred_logits: Final layer predictions (B, N, C)
            - pred_boxes: Final layer predictions (B, N, 4)
            - pred_corners: FGL head output (B, N, 4, reg_max+1)
            - ref_points: Reference points for FGL (B, N, 2)
            - aux_outputs: List of dicts from decoder intermediate layers
            - enc_aux_outputs: List of dicts from encoder intermediate layers
            - pre_outputs: Outputs before final refinement
        """
        B = images.shape[0]
        device = images.device
        
        # ===== Backbone =====
        _, backbone_feat = self.backbone(images)  # (B, 96, H', W')
        
        # ===== Neck (Multi-level features) =====
        multi_level_feats = self.neck(backbone_feat)  # List[(B, d_model, H_i, W_i)]
        
        # For simplicity, use only the first level (full resolution)
        # In advanced version,could process all levels separately
        feat = multi_level_feats[0]  # (B, d_model, H, W)
        
        # ===== Flatten features and compute positional encoding =====
        src = feat.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        pos = self.pe(feat)                    # (B, H*W, d_model)
        
        # ===== Encoder with Auxiliary Outputs =====
        memory = src
        enc_aux_outputs = []
        
        for enc_idx, enc_layer in enumerate(self.enc_layers):
            memory = enc_layer(memory, pos)
            
            # Generate auxiliary prediction heads for encoder outputs
            enc_logits = self.enc_aux_heads[enc_idx]['cls'](memory)  # (B, H*W, C)
            # For encoder aux, we average pool or take a representative query
            # This is simplified - in practice might project differently
            enc_aux_out = {
                'pred_logits': enc_logits.mean(dim=1, keepdim=True).expand(B, self.num_queries, -1),
                'pred_boxes': self.enc_aux_heads[enc_idx]['box'](memory).mean(dim=1, keepdim=True).expand(
                    B, self.num_queries, -1
                ),
            }
            enc_aux_outputs.append(enc_aux_out)
        
        # ===== Decoder with Auxiliary Outputs =====
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # (B, num_queries, d_model)
        qpos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)    # (B, num_queries, d_model)
        
        aux_outputs = []
        dec_out = tgt
        
        for dec_idx, dec_layer in enumerate(self.dec_layers):
            dec_out = dec_layer(dec_out, memory, qpos, pos)
            
            # Generate auxiliary predictions
            aux_logits = self.dec_aux_heads[dec_idx]['cls'](dec_out)
            aux_boxes = self.dec_aux_heads[dec_idx]['box'](dec_out)
            aux_corners = self.dec_aux_heads[dec_idx]['fgl'](dec_out)
            
            aux_output = {
                'pred_logits': aux_logits,
                'pred_boxes': F.sigmoid(aux_boxes),
                'pred_corners': aux_corners.view(B, self.num_queries, 4, self.reg_max + 1),
                'ref_points': self._generate_ref_points(self.num_queries, device),
            }
            aux_outputs.append(aux_output)
        
        # ===== Final Predictions =====
        pred_logits = self.cls_head(dec_out)  # (B, num_queries, num_classes)
        pred_boxes = self.box_head(dec_out)   # (B, num_queries, 4)
        pred_corners = self.fgl_head(dec_out) # (B, num_queries, 4*(reg_max+1))
        
        # Reshape FGL output
        pred_corners = pred_corners.view(B, self.num_queries, 4, self.reg_max + 1)
        
        # ===== Generate Reference Points =====
        ref_points = self._generate_ref_points(self.num_queries, device)  # (num_queries, 2)
        ref_points = ref_points.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, 2)
        
        # ===== Prepare Output Dict =====
        outputs = {
            'pred_logits': pred_logits,
            'pred_boxes': F.sigmoid(pred_boxes),
            'pred_corners': pred_corners,
            'ref_points': ref_points,
            'aux_outputs': aux_outputs[:-1],        # All but final
            'pre_outputs': aux_outputs[0] if aux_outputs else None,  # First decoder layer
            'enc_aux_outputs': enc_aux_outputs,
        }
        
        return outputs
    
    def _generate_ref_points(self, num_queries, device):
        """Generate reference points for FGL.
        
        In a full implementation, these would correspond to actual spatial locations.
        Here we generate a simple grid of reference points.
        
        Args:
            num_queries: Number of queries
            device: Device to create tensors on
        
        Returns:
            ref_points: (num_queries, 2) with normalized coordinates in [0, 1]
        """
        # Simple grid of reference points
        side = int(math.sqrt(num_queries))
        if side * side != num_queries:
            side = int(math.sqrt(num_queries)) + 1
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, side, device=device),
            torch.linspace(0, 1, side, device=device),
            indexing='ij'
        )
        ref_points = torch.stack([grid_x, grid_y], dim=-1).flatten(0, 1)[:num_queries]
        
        return ref_points
