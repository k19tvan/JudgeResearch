import torch
from starter import SetCriterion


def run_tests():
    print("Testing Set Criterion...")
    criterion = SetCriterion(num_classes=80)

    B, N, C = 2, 10, 80
    outputs = {
        "pred_logits": torch.randn(B, N, C),
        "pred_boxes":  torch.rand(B, N, 4).clamp(0.05, 0.95),
    }
    targets = [
        {"labels": torch.tensor([3, 7]),  "boxes": torch.rand(2, 4).clamp(0.05, 0.95)},
        {"labels": torch.tensor([15]),    "boxes": torch.rand(1, 4).clamp(0.05, 0.95)},
    ]
    indices = [
        (torch.tensor([1, 4]), torch.tensor([0, 1])),
        (torch.tensor([7]),    torch.tensor([0])),
    ]

    losses = criterion(outputs, targets, indices)

    # Test 1: All expected keys present
    assert "loss_vfl"  in losses, "Missing loss_vfl"
    assert "loss_bbox" in losses, "Missing loss_bbox"
    assert "loss_giou" in losses, "Missing loss_giou"

    # Test 2: All losses positive scalars
    for name, val in losses.items():
        assert val.shape == (), f"{name} must be scalar, got {val.shape}"
        assert val.item() >= 0, f"{name} must be non-negative, got {val.item()}"

    # Test 3: Losses must be finite
    for name, val in losses.items():
        assert torch.isfinite(val), f"{name} is not finite: {val}"

    # Test 4: Gradient flows through all losses (use require_grad=True inputs)
    pred_l = torch.randn(1, 5, 80, requires_grad=True)
    pred_b_raw = torch.rand(1, 5, 4) * 0.9 + 0.05  # safe values in [0.05, 0.95]
    pred_b = pred_b_raw.requires_grad_(True)          # proper leaf tensor
    out = {"pred_logits": pred_l, "pred_boxes": pred_b}
    tgt = [{"labels": torch.tensor([0]), "boxes": torch.rand(1, 4) * 0.5 + 0.25}]
    idx = [(torch.tensor([2]), torch.tensor([0]))]
    l2 = criterion(out, tgt, idx)
    sum(l2.values()).backward()
    assert pred_l.grad is not None, "Gradients must flow to pred_logits"
    assert pred_b.grad is not None, "Gradients must flow to pred_boxes"

    print("All Problem 05 checks passed")


if __name__ == "__main__":
    run_tests()
