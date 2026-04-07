import torch
from starter import box_iou, generalized_box_iou


def run_tests():
    print("Testing GIoU...")

    # Test 1: Perfect overlap → IoU=1, GIoU=1
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    iou, _ = box_iou(b, b)
    assert abs(iou[0, 0].item() - 1.0) < 1e-5, f"Perfect overlap IoU should be 1, got {iou[0,0]}"
    giou = generalized_box_iou(b, b)
    assert abs(giou[0, 0].item() - 1.0) < 1e-5, f"Perfect overlap GIoU should be 1, got {giou[0,0]}"

    # Test 2: No overlap → IoU=0, GIoU<0
    a = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    c = torch.tensor([[0.8, 0.8, 1.0, 1.0]])
    iou, _ = box_iou(a, c)
    assert iou[0, 0].item() == 0.0, "Non-overlapping boxes must have IoU=0"
    giou = generalized_box_iou(a, c)
    assert giou[0, 0].item() < 0.0, "Non-overlapping boxes must have GIoU<0"

    # Test 3: Partial overlap — manual value
    boxes1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    boxes2 = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
    iou, union = box_iou(boxes1, boxes2)
    assert abs(iou[0, 0].item() - (0.0625 / 0.4375)) < 1e-4, "Partial overlap IoU mismatch"

    # Test 4: Output shape for batch inputs
    N, M = 5, 7
    b1 = torch.rand(N, 4)
    b1[:, 2:] = b1[:, :2] + torch.rand(N, 2)
    b2 = torch.rand(M, 4)
    b2[:, 2:] = b2[:, :2] + torch.rand(M, 2)
    giou = generalized_box_iou(b1, b2)
    assert giou.shape == (N, M), f"GIoU output shape mismatch: {giou.shape}"

    # Test 5: GIoU always <= IoU, always >= -1
    assert (giou <= iou.expand_as(giou[0:1]) + 1.01).all()
    assert (giou >= -1.0 - 1e-5).all(), "GIoU must be >= -1"

    print("All Problem 02 checks passed")


if __name__ == "__main__":
    run_tests()
