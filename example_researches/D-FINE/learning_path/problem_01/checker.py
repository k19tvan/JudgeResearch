import torch
from starter import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

def run_tests():
    print("Testing Bounding Box Formats...")

    # Edge Case: Extreme bounds check
    cxcywh_base = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)
    expected_full = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    
    out_xyxy = box_cxcywh_to_xyxy(cxcywh_base)
    assert out_xyxy.shape == (1, 4), "Output shape strictly mutated!"
    assert torch.allclose(out_xyxy, expected_full), f"Conversion error on Base Bounds. Got {out_xyxy}"

    out_revert = box_xyxy_to_cxcywh(expected_full)
    assert torch.allclose(out_revert, cxcywh_base), "Reversion mapping distorted."

    # Batch test with diverse dimensional shape (B, N, 4)
    cxcywh_batch = torch.tensor([
        [[0.5, 0.5, 0.2, 0.4], [0.1, 0.2, 0.1, 0.1]],
        [[0.2, 0.2, 0.4, 0.4], [0.8, 0.8, 0.2, 0.2]]
    ], dtype=torch.float32)
    
    xyxy_batch = box_cxcywh_to_xyxy(cxcywh_batch)
    assert xyxy_batch.shape == (2, 2, 4), "Failed tensor graph integrity in batch sequence (B, N, 4)."
    
    cxcywh_test_revert = box_xyxy_to_cxcywh(xyxy_batch)
    assert torch.allclose(cxcywh_batch, cxcywh_test_revert, atol=1e-5), "Irregular float math during chained reversion mapping."
    
    print("All Problem 01 checks passed")

if __name__ == "__main__":
    run_tests()
