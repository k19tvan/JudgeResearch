import torch
from solution import DFINEMini, HungarianMatcher, SetCriterion, make_synthetic_batch, train_one_epoch, evaluate


def run_tests():
    print("Testing Train/Eval Loop (Phase 4 Verification)...")
    device = torch.device("cpu")
    NC, NQ = 10, 50

    model     = DFINEMini(nc=NC, nq=NQ, d=128, nel=1, ndl=2).to(device)
    matcher   = HungarianMatcher()
    criterion = SetCriterion(nc=NC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Test 1: make_synthetic_batch shape and validity
    imgs, tgts = make_synthetic_batch(2, 3, NC, device)
    assert imgs.shape == (2, 3, 256, 256), f"Image shape: {imgs.shape}"
    assert len(tgts) == 2
    for t in tgts:
        assert t["boxes"].shape[-1] == 4
        assert (t["boxes"][:, 2:] > 0).all(), "Widths/heights must be positive"
        assert t["boxes"].max() <= 1.0 + 1e-4 and t["boxes"].min() >= 0 - 1e-4

    # Test 2: Single forward + backward pass (smoke check)
    loader = [make_synthetic_batch(2, 3, NC, device) for _ in range(3)]
    imgs, tgts = loader[0]
    model.train()
    out = model(imgs.to(device))
    indices = matcher(out, [{k:v.to(device) for k,v in t.items()} for t in tgts])["indices"]
    losses = criterion(out, [{k:v.to(device) for k,v in t.items()} for t in tgts], indices)
    total = sum(losses.values())
    assert torch.isfinite(total), f"Loss is not finite: {total}"
    total.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed!"

    # Test 3: Short training run (10 iterations) — loss must be finite throughout
    loader_train = [make_synthetic_batch(2, 2, NC, device) for _ in range(10)]
    train_metrics = train_one_epoch(model, criterion, matcher, optimizer, loader_train, device)
    assert "avg_loss_vfl"  in train_metrics
    assert "avg_loss_bbox" in train_metrics
    assert "avg_loss_giou" in train_metrics
    for k, v in train_metrics.items():
        assert isinstance(v, float) and v >= 0 and not (v != v), f"{k} is not finite: {v}"

    # Test 4: Evaluate (inference) runs without errors
    loader_eval = [make_synthetic_batch(2, 2, NC, device) for _ in range(3)]
    eval_metrics = evaluate(model, loader_eval, device)
    assert "mean_iou" in eval_metrics
    assert 0.0 <= eval_metrics["mean_iou"] <= 1.0 + 1e-4, f"mean_iou out of range: {eval_metrics['mean_iou']}"

    print("All Problem 12 checks passed")
    print(f"  Train metrics: {train_metrics}")
    print(f"  Eval metrics:  {eval_metrics}")
    print("\n=== ALL 12 PROBLEMS VERIFIED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_tests()
