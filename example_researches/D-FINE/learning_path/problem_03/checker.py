import torch
from starter import HungarianMatcher


def run_tests():
    print("Testing Hungarian Matcher...")
    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    # Test 1: Basic — B=1, N=5, T=2
    outputs = {
        "pred_logits": torch.randn(1, 5, 80),
        "pred_boxes":  torch.rand(1, 5, 4).clamp(0.01, 0.99),
    }
    targets = [{"labels": torch.tensor([3, 7]), "boxes": torch.rand(2, 4).clamp(0.01, 0.99)}]
    result = matcher(outputs, targets)
    assert "indices" in result, "Result must contain 'indices'"
    indices = result["indices"]
    assert len(indices) == 1, "Must return one assignment per image"
    src_idx, tgt_idx = indices[0]
    assert len(src_idx) == 2, "Must match exactly T=2 predictions"
    assert len(tgt_idx) == 2, "Must match exactly T=2 targets"
    assert src_idx.max() < 5, "src_idx must be in [0, N)"
    assert tgt_idx.max() < 2, "tgt_idx must be in [0, T)"
    assert len(set(src_idx.tolist())) == 2, "src_idx must be unique (1-to-1)"
    assert len(set(tgt_idx.tolist())) == 2, "tgt_idx must be unique (1-to-1)"

    # Test 2: Batch — B=2 with different T counts
    outputs2 = {
        "pred_logits": torch.randn(2, 10, 80),
        "pred_boxes":  torch.rand(2, 10, 4).clamp(0.01, 0.99),
    }
    targets2 = [
        {"labels": torch.tensor([0]),          "boxes": torch.rand(1, 4)},
        {"labels": torch.tensor([1, 2, 5]),    "boxes": torch.rand(3, 4)},
    ]
    result2 = matcher(outputs2, targets2)
    assert len(result2["indices"]) == 2
    assert len(result2["indices"][0][0]) == 1   # image 0: 1 gt
    assert len(result2["indices"][1][0]) == 3   # image 1: 3 gts

    # Test 3: Edge — T=0 (empty ground truth)
    outputs3 = {"pred_logits": torch.randn(1, 5, 80), "pred_boxes": torch.rand(1, 5, 4)}
    targets3 = [{"labels": torch.zeros(0, dtype=torch.int64), "boxes": torch.zeros(0, 4)}]
    result3 = matcher(outputs3, targets3)
    assert len(result3["indices"][0][0]) == 0, "Empty GT must yield empty assignment"

    print("All Problem 03 checks passed")


if __name__ == "__main__":
    run_tests()
