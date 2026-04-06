import torch
from starter import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

def run_checks():
    # Test 1: Core correctness (Basic 2D batch)
    cxcywh_input = torch.tensor([
        [0.5, 0.5, 1.0, 1.0],  # Centered unit box
        [10.0, 20.0, 4.0, 6.0] # Arbitrary box
    ], dtype=torch.float32)
    
    expected_xyxy = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [8.0, 17.0, 12.0, 23.0]
    ], dtype=torch.float32)
    
    # Test cxcywh_to_xyxy
    out_xyxy = box_cxcywh_to_xyxy(cxcywh_input)
    assert out_xyxy.shape == (2, 4), f"Shape mismatch: {out_xyxy.shape}"
    assert torch.allclose(out_xyxy, expected_xyxy), "cxcywh_to_xyxy failed."
    
    # Test xyxy_to_cxcywh
    out_cxcywh = box_xyxy_to_cxcywh(expected_xyxy)
    assert out_cxcywh.shape == (2, 4), f"Shape mismatch: {out_cxcywh.shape}"
    assert torch.allclose(out_cxcywh, cxcywh_input), "xyxy_to_cxcywh failed."
    
    # Test 2: Edge Case (Higher dimensional tensors, e.g., Batch x Sequence x 4)
    batch_cxcywh = torch.zeros((4, 300, 4))
    batch_cxcywh[..., 0:2] = 50.0 # centers at 50
    batch_cxcywh[..., 2:4] = 20.0 # width/height 20
    
    expected_batch_xyxy = torch.zeros((4, 300, 4))
    expected_batch_xyxy[..., 0:2] = 40.0 # 50 - 10
    expected_batch_xyxy[..., 2:4] = 60.0 # 50 + 10
    
    out_batch = box_cxcywh_to_xyxy(batch_cxcywh)
    assert out_batch.shape == (4, 300, 4), f"Shape mismatch on 3D tensor: {out_batch.shape}"
    assert torch.allclose(out_batch, expected_batch_xyxy), "Multi-dimensional handling failed."

    print("All Problem 01 checks passed")

if __name__ == "__main__":
    run_checks()
