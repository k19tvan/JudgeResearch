import torch
from starter import sigmoid_focal_loss, varifocal_loss


def run_tests():
    print("Testing Focal / VFL Losses...")

    C = 4

    # Test 1: Focal loss returns scalar
    logits = torch.randn(10, C)
    targets = torch.zeros(10, C)
    targets[:5, 0] = 1.0
    loss = sigmoid_focal_loss(logits, targets)
    assert loss.shape == (), f"Focal loss must be a scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Focal loss must be positive"

    # Test 2: Focal loss with perfect prediction should be near 0
    large_logit = torch.full((5, C), -10.0)
    large_logit[:, 0] = 10.0
    targets_p = torch.zeros(5, C)
    targets_p[:, 0] = 1.0
    perfect_loss = sigmoid_focal_loss(large_logit, targets_p)
    assert perfect_loss.item() < 0.01, f"Perfect focal loss should be ~0, got {perfect_loss}"

    # Test 3: VFL output shape
    N = 8
    pred = torch.randn(N, C)
    gt_score = torch.rand(N)
    label = torch.randint(0, C, (N,))
    vfl = varifocal_loss(pred, gt_score, label, num_classes=C)
    assert vfl.shape == (N, C), f"VFL shape mismatch: {vfl.shape}"
    assert (vfl >= 0).all(), "VFL elements must be non-negative"

    # Test 4: VFL positive class loss decreases as gt_score→1 and logit→high
    pred_perfect = torch.zeros(1, C)
    pred_perfect[0, 0] = 10.0
    vfl_perfect = varifocal_loss(pred_perfect, torch.tensor([0.99]), torch.tensor([0]), C)
    pred_bad = torch.zeros(1, C)
    pred_bad[0, 0] = -10.0
    vfl_bad = varifocal_loss(pred_bad, torch.tensor([0.99]), torch.tensor([0]), C)
    assert vfl_perfect[0, 0].item() < vfl_bad[0, 0].item(), \
        "Better prediction should have lower VFL loss on positive class"

    print("All Problem 04 checks passed")


if __name__ == "__main__":
    run_tests()
