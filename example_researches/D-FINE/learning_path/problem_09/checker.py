import torch
from starter import TransformerDecoderLayer


def run_tests():
    print("Testing Transformer Decoder Layer (with Target Gating)...")
    B, N, T_mem, d = 2, 10, 49, 64
    layer = TransformerDecoderLayer(d_model=d, num_heads=8, d_ffn=256, dropout=0.0)

    tgt    = torch.randn(B, N, d)
    memory = torch.randn(B, T_mem, d)
    qpos   = torch.randn(B, N, d)
    mpos   = torch.randn(B, T_mem, d)

    # Test 1: Output shape
    out = layer(tgt, memory, qpos, mpos)
    assert out.shape == (B, N, d), f"Output shape: {out.shape}"

    # Test 2: Finite values
    assert torch.isfinite(out).all(), "Output must be finite"

    # Test 3: Memory padding mask
    mask = torch.zeros(B, T_mem, dtype=torch.bool)
    mask[:, -5:] = True
    out_m = layer(tgt, memory, qpos, mpos, memory_key_padding_mask=mask)
    assert out_m.shape == (B, N, d)
    assert torch.isfinite(out_m).all()

    # Test 4: Gradient flows through gating
    tgt_g = tgt.clone().requires_grad_(True)
    out_g = layer(tgt_g, memory, qpos, mpos)
    out_g.sum().backward()
    assert tgt_g.grad is not None, "Gradient must flow through decoder layer"

    # Test 5: Memory gradient flows (cross-attention path)
    mem_g = memory.clone().requires_grad_(True)
    out_g2 = layer(tgt, mem_g, qpos, mpos)
    out_g2.sum().backward()
    assert mem_g.grad is not None, "Gradient must flow from memory through cross-attention"

    print("All Problem 09 checks passed")


if __name__ == "__main__":
    run_tests()
