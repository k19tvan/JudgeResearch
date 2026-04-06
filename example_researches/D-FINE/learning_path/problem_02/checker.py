import torch
from starter import box_area, box_iou, generalized_box_iou

def run_checks():
    # Setup test boxes
    boxes1 = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],  # 10x10 square
        [5.0, 5.0, 15.0, 15.0]   # 10x10 square offset
    ], dtype=torch.float32)
    
    boxes2 = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],      # Perfect match for boxes1[0]
        [20.0, 20.0, 30.0, 30.0],    # Disjoint box
        [2.5, 2.5, 7.5, 7.5]         # Inside boxes1[0]
    ], dtype=torch.float32)

    # Check box_area
    areas = box_area(boxes1)
    assert areas.shape == (2,), "box_area shape mismatch"
    assert torch.allclose(areas, torch.tensor([100.0, 100.0])), "box_area calculation failed"

    # Check box_iou
    ious, union = box_iou(boxes1, boxes2)
    assert ious.shape == (2, 3), "box_iou shape mismatch"
    assert ious[0, 0] == 1.0, "Perfect box should have IoU 1.0"
    assert ious[0, 1] == 0.0, "Disjoint boxes should have IoU 0.0"
    assert ious[0, 2] == 0.25, "Quarter size inside box should have IoU 0.25 (25 / 100)"

    # Check generalized_box_iou
    giou = generalized_box_iou(boxes1, boxes2)
    assert giou.shape == (2, 3), "giou shape mismatch"
    assert giou[0, 0] == 1.0, "Perfect box should have GIoU 1.0"
    
    # Check penalty for disjoint boxes (boxes1[0] vs boxes2[1] over a 30x30 minimal enclosing box -> 900 area)
    # union is 100 + 100 = 200, enclosing is 900. Penalty = -(900 - 200) / 900 = -700/900 = -0.7777...
    # giou should be 0 - 0.7777 = -0.7777
    assert giou[0, 1] < 0.0, "Disjoint boxes must have negative GIoU"
    assert round(giou[0,1].item(), 4) == -0.7778, f"GIoU penalty mathematically incorrect: {giou[0,1]}"

    print("All Problem 02 checks passed")

if __name__ == "__main__":
    run_checks()