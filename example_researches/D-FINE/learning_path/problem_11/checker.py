import torch
from starter import DFINEMini


def run_tests():
    print("Testing DFINEMini End-to-End Model...")
    B, NC, NQ, DIM = 2, 80, 100, 256
    model = DFINEMini(num_classes=NC, num_queries=NQ, d_model=DIM,
                      num_encoder_layers=2, num_decoder_layers=3).eval()

    images = torch.randn(B, 3, 256, 256)

    with torch.no_grad():
        out = model(images)

    # Test 1: Keys present
    assert "pred_logits" in out, "Missing pred_logits"
    assert "pred_boxes"  in out, "Missing pred_boxes"

    # Test 2: Shapes
    assert out["pred_logits"].shape == (B, NQ, NC), f"Logit shape: {out['pred_logits'].shape}"
    assert out["pred_boxes"].shape  == (B, NQ, 4),  f"Box shape: {out['pred_boxes'].shape}"

    # Test 3: Boxes in [0, 1]
    assert out["pred_boxes"].min().item() >= -1e-5, "Box values must be >= 0"
    assert out["pred_boxes"].max().item() <= 1+1e-5, "Box values must be <= 1"

    # Test 4: Finite outputs
    assert torch.isfinite(out["pred_logits"]).all(), "pred_logits has NaN/Inf"
    assert torch.isfinite(out["pred_boxes"]).all(),  "pred_boxes has NaN/Inf"

    # Test 5: Gradient End-to-End
    model.train()
    img_g = torch.randn(1, 3, 256, 256, requires_grad=True)
    out_g = model(img_g)
    loss = out_g["pred_logits"].sum() + out_g["pred_boxes"].sum()
    loss.backward()
    assert img_g.grad is not None, "Gradients must flow end-to-end from loss to image"

    print("All Problem 11 checks passed")


if __name__ == "__main__":
    run_tests()
