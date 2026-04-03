# learning_path/problem_01/checker.py
import torch
from starter import box_xywh_to_cxcxw, box_cxcxw_to_xyxy

def check_conversions():
    img_h, img_w = 400, 800
    # [x_min, y_min, w, h]
    boxes = torch.tensor([[100.0, 100.0, 200.0, 100.0]])
    
    out_cxcxw = box_xywh_to_cxcxw(boxes, (img_h, img_w))
    assert out_cxcxw.shape == (1, 4), f"Shape mismatch: {out_cxcxw.shape}"
    
    expected_cxcxw = torch.tensor([[0.25, 0.375, 0.25, 0.25]])
    assert torch.allclose(out_cxcxw, expected_cxcxw), f"cxcxw Math error: {out_cxcxw}"
    
    out_xyxy = box_cxcxw_to_xyxy(out_cxcxw, (img_h, img_w))
    expected_xyxy = torch.tensor([[100.0, 100.0, 300.0, 200.0]])
    assert torch.allclose(out_xyxy, expected_xyxy), f"xyxy Math error: {out_xyxy}"
    
    print("All Problem 01 checks passed")

if __name__ == "__main__":
    check_conversions()
