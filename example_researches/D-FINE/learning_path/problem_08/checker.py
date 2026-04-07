import torch
from starter import TransformerEncoderLayer


def run_tests():
    print("Testing Transformer Encoder Layer...")
    B, T, d = 2, 49, 128
    layer = TransformerEncoderLayer(d_model=d, num_heads=8, d_ffn=512, dropout=0.0)

    src = torch.randn(B, T, d)
    pos = torch.randn(B, T, d)

    # Test 1: Output shape
    out = layer(src, pos)
    assert out.shape == (B, T, d), f"Output shape: {out.shape}"

    # Test 2: With padding mask
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, -3:] = True
    out_m = layer(src, pos, src_key_padding_mask=mask)
    assert out_m.shape == (B, T, d)

    # Test 3: Finite output
    assert torch.isfinite(out).all(), "Output must be finite"

    # Test 4: residual connection keeps input scale
    with torch.no_grad():
        ratio = out.norm() / src.norm()
    assert 0.1 < ratio.item() < 10, f"Output scale changed dramatically: ratio={ratio}"

    # Test 5: Gradient flows end-to-end
    src_g = src.clone().requires_grad_(True)
    out_g = layer(src_g, pos)
    out_g.sum().backward()
    assert src_g.grad is not None, "Gradient must flow to src"

    print("All Problem 08 checks passed")


if __name__ == "__main__":
    run_tests()
