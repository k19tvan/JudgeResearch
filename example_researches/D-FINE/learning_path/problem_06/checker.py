import torch
from starter import PositionEmbeddingSine2D


def run_tests():
    print("Testing Sine 2D Positional Encoding...")
    d = 256
    pe = PositionEmbeddingSine2D(d_model=d, temperature=10000)

    # Test 1: Output shape
    B, C, H, W = 2, 512, 7, 7
    x = torch.zeros(B, C, H, W)
    enc = pe(x)
    assert enc.shape == (B, H*W, d), f"Shape mismatch: {enc.shape}"

    # Test 2: Values in [-1, 1]
    assert enc.abs().max().item() <= 1.0 + 1e-5, f"Encoding values exceed [-1,1]: {enc.abs().max()}"

    # Test 3: No learnable parameters
    params = list(pe.parameters())
    assert len(params) == 0, "PositionEmbeddingSine2D must have no learnable parameters"

    # Test 4: Different (H, W) still works (arbitrary spatial size)
    x2 = torch.zeros(1, 256, 13, 17)
    enc2 = pe(x2)
    assert enc2.shape == (1, 13*17, d), f"Arbitrary HxW failed: {enc2.shape}"

    # Test 5: Every spatial position gets a different encoding
    enc_flat = enc[0]  # (H*W, d_model)
    # pairwise L2 distances between all position encodings
    diff = (enc_flat.unsqueeze(0) - enc_flat.unsqueeze(1)).norm(dim=-1)  # (H*W, H*W)
    diag_zero = diff.diag()
    assert (diag_zero < 1e-5).all(), "Same-position encodings should be identical"
    off_diag = diff + torch.eye(H*W) * 100  # mask diagonal
    assert (off_diag.min() > 0.1), "Different positions must have different encodings"

    print("All Problem 06 checks passed")


if __name__ == "__main__":
    run_tests()
