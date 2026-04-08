"""Problem 14: Checker - Validate FGL Loss Implementation"""
import torch
from solution import bbox2distance, unimodal_distribution_focal_loss


def test_bbox2distance_basic():
    \"\"\"Test basic distance calculation.\"\"\"
    # Simple case: box and reference point
    points = torch.tensor([[100.0, 130.0]], dtype=torch.float32)
    bbox = torch.tensor([[50.0, 60.0, 150.0, 200.0]], dtype=torch.float32)
    
    distances, soft_label, weight = bbox2distance(points, bbox, reg_max=32)
    
    # Check distances
    assert distances.shape == (1, 4), f"Expected (1, 4), got {distances.shape}"
    expected_dist = torch.tensor([[50.0, 50.0, 70.0, 70.0]])  # Before clamping
    # After clamping to reg_max=32: [32, 32, 32, 32]
    assert torch.allclose(distances, torch.clamp(expected_dist, max=32).float()), \
        f"Distances don't match. Got {distances}"
    
    print("✓ test_bbox2distance_basic passed")


def test_soft_labels_properties():
    \"\"\"Test soft label properties.\"\"\"
    points = torch.tensor([[100.0, 130.0]], dtype=torch.float32)
    bbox = torch.tensor([[50.0, 60.0, 150.0, 200.0]], dtype=torch.float32)
    
    distances, soft_label, weight = bbox2distance(points, bbox, reg_max=32)
    
    # Check shape
    assert soft_label.shape == (1, 4, 33), f"Expected (1, 4, 33), got {soft_label.shape}"
    
    # Check that soft labels are valid distributions
    # Each (distance, side) should sum to approximately 1.0 (with interpolation)
    label_sums = soft_label.sum(dim=-1)
    assert torch.allclose(label_sums, torch.ones_like(label_sums), atol=1e-5), \
        f\"Soft labels don't sum to 1: {label_sums}\"
    
    # Check values are in [0, 1]
    assert (soft_label >= 0).all() and (soft_label <= 1).all(), \
        \"Soft label values must be in [0, 1]\"
    
    print("✓ test_soft_labels_properties passed")


def test_batch_processing():
    \"\"\"Test handling of batch inputs.\"\"\"
    batch_size = 8
    points = torch.randn(batch_size, 2) + 100  # Random points near [100, 100]
    bbox = torch.cat([
        torch.randn(batch_size, 2) * 30 + 50,   # x1, y1
        torch.randn(batch_size, 2) * 30 + 150   # x2, y2
    ], dim=1)
    bbox = bbox.clamp(min=0, max=200)  # Clamp to valid range
    
    distances, soft_label, weight = bbox2distance(points, bbox, reg_max=32)
    
    assert distances.shape == (batch_size, 4)
    assert soft_label.shape == (batch_size, 4, 33)
    assert weight.shape == (batch_size, 4)
    
    print("✓ test_batch_processing passed")


def test_out_of_range_handling():
    \"\"\"Test handling of out-of-range distances.\"\"\"
    points = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    bbox = torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32)  # Very far
    
    distances, soft_label, weight = bbox2distance(points, bbox, reg_max=32)
    
    # Distances should be clamped to 32
    assert torch.all(distances <= 32), f\"Distances exceed reg_max: {distances}\"
    
    # Out-of-range distances should have reduced weight
    assert torch.all(weight < 1.0), \"Out-of-range should have weight < 1.0\"
    
    print("✓ test_out_of_range_handling passed")


def test_focal_loss_basic():
    \"\"\"Test focal loss computation.\"\"\"
    batch_size, num_sides, bins = 4, 4, 33
    
    # Create mock predictions (logits)
    pred_dist = torch.randn(batch_size, num_sides, bins)
    
    # Create mock targets (soft labels)
    soft_label = torch.zeros(batch_size, num_sides, bins)
    # Set a soft label with peak at bin 5
    for i in range(batch_size):
        for j in range(num_sides):
            soft_label[i, j, 5] = 0.7
            soft_label[i, j, 6] = 0.3
    
    loss = unimodal_distribution_focal_loss(pred_dist, soft_label, reduction='mean')
    
    # Check loss is scalar and numeric
    assert loss.dim() == 0, f\"Expected scalar loss, got shape {loss.shape}\"
    assert torch.isfinite(loss), f\"Loss is not finite: {loss}\"
    assert loss.item() > 0, f\"Loss should be positive, got {loss.item()}\"
    
    print("✓ test_focal_loss_basic passed")


def test_focal_loss_perfect_prediction():
    \"\"\"Test that perfect predictions have low loss.\"\"\"
    # Perfect prediction: pred_dist matches soft_label distribution
    soft_label = torch.zeros(2, 4, 33)
    soft_label[:, :, 10] = 1.0  # All probability on bin 10
    
    # Make perfect prediction by setting logits to match
    pred_dist = torch.zeros(2, 4, 33)
    pred_dist[:, :, 10] = 10.0  # High logit for correct bin
    
    loss = unimodal_distribution_focal_loss(pred_dist, soft_label, reduction='mean')
    
    # Loss should be very low (but not exactly 0 due to softmax)
    assert loss.item() < 0.1, f\"Perfect prediction should have low loss, got {loss.item()}\"
    
    print("✓ test_focal_loss_perfect_prediction passed")


def test_focal_loss_random_prediction():
    \"\"\"Test that random predictions have reasonable loss.\"\"\"
    soft_label = torch.zeros(4, 4, 33)
    # Create varied targets
    for i in range(4):
        for j in range(4):
            bin_idx = (i * 4 + j) % 32
            soft_label[i, j, bin_idx] = 1.0
    
    # Random predictions
    torch.manual_seed(42)
    pred_dist = torch.randn(4, 4, 33)
    
    loss = unimodal_distribution_focal_loss(pred_dist, soft_label, reduction='mean')
    
    assert 0 < loss.item() < 1000, f\"Loss out of reasonable range: {loss.item()}\"
    
    print(\"✓ test_focal_loss_random_prediction passed\")


def test_weight_application():
    \"\"\"Test that weight parameter scales loss.\"\"\"
    soft_label = torch.ones(2, 4, 33) / 33  # Uniform distribution
    pred_dist = torch.randn(2, 4, 33)
    
    # Compute loss without weights
    loss_no_weight = unimodal_distribution_focal_loss(
        pred_dist, soft_label, weight=None, reduction='mean'
    )
    
    # Compute loss with all weights = 0.5
    weights = torch.ones(2, 4) * 0.5
    loss_with_weight = unimodal_distribution_focal_loss(
        pred_dist, soft_label, weight=weights, reduction='mean'
    )
    
    # Weighted loss should be smaller
    assert loss_with_weight < loss_no_weight, \
        f\"Weighted loss {loss_with_weight} should be < unweighted {loss_no_weight}\"
    
    # Weighted loss should be approximately 0.5x
    ratio = loss_with_weight / loss_no_weight
    assert 0.4 < ratio < 0.6, f\"Weight scaling incorrect: {ratio}\"
    
    print(\"✓ test_weight_application passed\")


if __name__ == \"__main__\":
    test_bbox2distance_basic()
    test_soft_labels_properties()
    test_batch_processing()
    test_out_of_range_handling()
    test_focal_loss_basic()
    test_focal_loss_perfect_prediction()
    test_focal_loss_random_prediction()
    test_weight_application()
    
    print(\"\\n\" + \"=\"*50)
    print(\"All tests passed! ✓\")
    print(\"=\"*50)
