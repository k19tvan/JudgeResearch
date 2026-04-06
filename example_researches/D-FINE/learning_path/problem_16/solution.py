import torch
import torch.nn as nn

class EndToEndDFINE(nn.Module):
    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.backbone(x)
        encoded_feats = self.encoder(features)
        out = self.decoder(encoded_feats)
        return out
