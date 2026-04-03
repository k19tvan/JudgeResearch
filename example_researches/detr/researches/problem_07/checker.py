# learning_path/problem_07/checker.py
import torch
import torch.nn as nn
from starter import DETR

class MockBackbone(nn.Module):
    def forward(self, x):
        return torch.rand(x.shape[0], 256, x.shape[2]//32, x.shape[3]//32)

class MockTransformer(nn.Module):
    def forward(self, src, mask, query_embed, pos_embed):
        B = src.shape[1] if len(src.shape) == 3 else src.shape[0]
        N = query_embed.shape[0]
        return torch.rand(1, B, N, 256).squeeze(0)

def check_detr():
    B, N, C = 2, 100, 256
    num_classes = 91
    
    model = DETR(MockBackbone(), MockTransformer(), num_classes, N, C)
    x = torch.rand(B, 3, 800, 800)
    
    try:
        out = model(x)
        assert 'pred_logits' in out
        assert 'pred_boxes' in out
        assert out['pred_logits'].shape == (B, N, num_classes + 1)
        assert out['pred_boxes'].shape == (B, N, 4)
        print("All Problem 07 checks passed")
    except NotImplementedError:
        print("Implement the DETR component assembly to pass.")
        raise

if __name__ == "__main__":
    check_detr()
