import torch
from starter import MultiHeadAttention


def run_tests():
    print("Testing Multi-Head Attention...")
    B, T_q, T_k, d, H = 2, 10, 20, 64, 8
    mha = MultiHeadAttention(d_model=d, num_heads=H, dropout=0.0)

    q = torch.randn(B, T_q, d)
    k = torch.randn(B, T_k, d)
    v = torch.randn(B, T_k, d)

    # Test 1: Output shapes
    out, attn = mha(q, k, v)
    assert out.shape == (B, T_q, d), f"Output shape: {out.shape}"
    assert attn.shape == (B, H, T_q, T_k), f"Attention shape: {attn.shape}"

    # Test 2: Attention weights sum to 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, H, T_q), atol=1e-5), \
        "Attention weights must sum to 1 over T_k"

    # Test 3: Attention weights non-negative
    assert (attn >= 0).all(), "Attention weights must be non-negative"

    # Test 4: Key padding mask zeros out masked positions
    mask = torch.zeros(B, T_k, dtype=torch.bool)
    mask[:, -5:] = True   # mask last 5 positions
    out_m, attn_m = mha(q, k, v, key_padding_mask=mask)
    assert (attn_m[:, :, :, -5:].abs() < 1e-5).all(), "Masked positions must have ~0 attention"

    # Test 5: Self-attention (q=k=v) produces finite output
    sa_out, _ = mha(q, q, q)
    assert torch.isfinite(sa_out).all(), "Self-attention output must be finite"

    # Test 6: Gradient flows
    q_g = q.requires_grad_(True)
    out_g, _ = mha(q_g, k, v)
    out_g.sum().backward()
    assert q_g.grad is not None, "Gradient must flow to q"

    print("All Problem 07 checks passed")


if __name__ == "__main__":
    run_tests()
