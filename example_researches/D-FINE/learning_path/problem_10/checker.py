import torch
from starter import HGNetV2Stem


def run_tests():
    print("Testing HGNetV2-S Stem + LCS Block...")
    net = HGNetV2Stem().eval()

    # Test 1: Standard shapes
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        s_out, l_out = net(x)
    assert s_out.shape == (2, 48, 64, 64),  f"Stem out shape: {s_out.shape}"
    assert l_out.shape == (2, 96, 32, 32),  f"LCS out shape: {l_out.shape}"

    # Test 2: Different input size
    x2 = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        s2, l2 = net(x2)
    assert s2.shape == (1, 48, 160, 160)
    assert l2.shape == (1, 96, 80, 80)

    # Test 3: Finite outputs
    assert torch.isfinite(s_out).all(), "Stem output has NaN/Inf"
    assert torch.isfinite(l_out).all(), "LCS output has NaN/Inf"

    # Test 4: No NaN even with negative pixel values
    x_neg = torch.randn(1, 3, 128, 128) * 10
    with torch.no_grad():
        s_neg, l_neg = net(x_neg)
    assert torch.isfinite(l_neg).all()

    # Test 5: Gradient flows (train mode)
    net.train()
    x_g = torch.randn(1, 3, 128, 128, requires_grad=True)
    s_g, l_g = net(x_g)
    l_g.sum().backward()
    assert x_g.grad is not None, "Gradient must flow to input"

    print("All Problem 10 checks passed")


if __name__ == "__main__":
    run_tests()
