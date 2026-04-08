"""Problem 15: Checker - Validate Multi-Layer D-FINE Criterion"""
import torch
import torch.nn as nn
from solution import DFINECriterion


class MockMatcher:
    """Mock matcher for testing criterion."""
    
    def __call__(self, outputs, targets):
        """Return mock indices between predictions and targets."""
        batch_size = len(targets)
        indices = []
        
        for batch_idx in range(batch_size):
            num_queries = outputs["pred_logits"].shape[1]
            num_targets = len(targets[batch_idx]["labels"])
            
            if num_targets == 0:
                # Empty targets
                indices.append((torch.tensor([], dtype=torch.long),
                               torch.tensor([], dtype=torch.long)))
            else:
                # Simple matching: match first min(num_queries, num_targets) predictions
                num_matches = min(num_queries, num_targets)
                src_idx = torch.arange(num_matches, dtype=torch.long)
                tgt_idx = torch.arange(num_matches, dtype=torch.long)
                indices.append((src_idx, tgt_idx))
        
        return indices
    
    def compute_matching_union(self, indices_list):
        """Compute consensus matching from list of indices."""
        if not indices_list:
            return indices_list
        
        # Simple implementation: use first layer's indices
        return indices_list[0]


def test_criterion_basic():
    """Test basic criterion creation and forward pass."""
    criterion = DFINECriterion(
        matcher=MockMatcher(),
        num_classes=80,
        weight_dict={"loss_vfl": 1.0, "loss_fgl": 5.0},
        losses=["vfl", "fgl"],
        num_layers=6,
        reg_max=32
    )
    
    # Create mock outputs
    batch_size, num_queries = 2, 100
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, 80),
        "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
        "pred_corners": torch.randn(batch_size, num_queries, 4*33),
        "aux_outputs": [
            {
                "pred_logits": torch.randn(batch_size, num_queries, 80),
                "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
                "pred_corners": torch.randn(batch_size, num_queries, 4*33),
            }
            for _ in range(5)
        ]
    }
    
    # Create mock targets
    targets = [
        {
            "labels": torch.randint(0, 80, (5,)),
            "boxes": torch.rand(5, 4),
        },
        {
            "labels": torch.randint(0, 80, (3,)),
            "boxes": torch.rand(3, 4),
        }
    ]
    
    # Forward pass
    losses = criterion(outputs, targets)
    
    # Validate output structure
    assert isinstance(losses, dict), "Output should be a dictionary"
    assert "loss_vfl" in losses, "Should have 'loss_vfl' key"
    assert "loss_fgl" in losses, "Should have 'loss_fgl' key"
    
    print("✓ test_criterion_basic passed")


def test_criterion_output_keys():
    """Test that all expected loss keys are present."""
    criterion = DFINECriterion(
        matcher=MockMatcher(),
        num_classes=80,
        weight_dict={"loss_vfl": 1.0, "loss_fgl": 5.0},
        losses=["vfl", "fgl"],
        num_layers=6,
        reg_max=32
    )
    
    batch_size, num_queries = 1, 50
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, 80),
        "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
        "pred_corners": torch.randn(batch_size, num_queries, 4*33),
        "aux_outputs": [
            {
                "pred_logits": torch.randn(batch_size, num_queries, 80),
                "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
                "pred_corners": torch.randn(batch_size, num_queries, 4*33),
            }
            for _ in range(3)  # 3 auxiliary layers
        ]
    }
    
    targets = [{
        "labels": torch.randint(0, 80, (2,)),
        "boxes": torch.rand(2, 4)
    }]
    
    losses = criterion(outputs, targets)
    
    # Check keys exist
    expected_keys = [
        "loss_vfl", "loss_fgl",  # Final layer
        "aux_0_loss_vfl", "aux_0_loss_fgl",  # Aux layer 0
        "aux_1_loss_vfl", "aux_1_loss_fgl",  # Aux layer 1
        "aux_2_loss_vfl", "aux_2_loss_fgl",  # Aux layer 2
    ]
    
    for key in expected_keys:
        assert key in losses, f"Missing key: {key}"
    
    print("✓ test_criterion_output_keys passed")


def test_criterion_loss_values():
    """Test that loss values are valid (scalar, finite, positive)."""
    criterion = DFINECriterion(
        matcher=MockMatcher(),
        num_classes=80,
        weight_dict={"loss_vfl": 1.0, "loss_fgl": 5.0},
        losses=["vfl", "fgl"],
        num_layers=6,
        reg_max=32
    )
    
    batch_size, num_queries = 2, 100
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, 80),
        "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
        "pred_corners": torch.randn(batch_size, num_queries, 4*33),
        "aux_outputs": [
            {
                "pred_logits": torch.randn(batch_size, num_queries, 80),
                "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
                "pred_corners": torch.randn(batch_size, num_queries, 4*33),
            }
            for _ in range(2)
        ]
    }
    
    targets = [
        {"labels": torch.randint(0, 80, (4,)), "boxes": torch.rand(4, 4)},
        {"labels": torch.randint(0, 80, (2,)), "boxes": torch.rand(2, 4)}
    ]
    
    losses = criterion(outputs, targets)
    
    for key, value in losses.items():
        # Check scalar
        assert value.dim() == 0, f"{key} should be scalar, got dim {value.dim()}"
        # Check finite
        assert torch.isfinite(value), f"{key} is not finite: {value}"
        # Check positive (can have very small losses)
        assert value.item() >= 0, f"{key} should be non-negative, got {value.item()}"
    
    print("✓ test_criterion_loss_values passed")


def test_criterion_gradient_flow():
    """Test that gradients flow through criterion."""
    criterion = DFINECriterion(
        matcher=MockMatcher(),
        num_classes=80,
        weight_dict={"loss_vfl": 1.0, "loss_fgl": 5.0},
        losses=["vfl", "fgl"],
        num_layers=6,
        reg_max=32
    )
    
    batch_size, num_queries = 1, 50
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, 80, requires_grad=True),
        "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4, requires_grad=True)),
        "pred_corners": torch.randn(batch_size, num_queries, 4*33, requires_grad=True),
        "aux_outputs": [
            {
                "pred_logits": torch.randn(batch_size, num_queries, 80, requires_grad=True),
                "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4, requires_grad=True)),
                "pred_corners": torch.randn(batch_size, num_queries, 4*33, requires_grad=True),
            }
            for _ in range(2)
        ]
    }
    
    targets = [{
        "labels": torch.randint(0, 80, (3,)),
        "boxes": torch.rand(3, 4)
    }]
    
    losses = criterion(outputs, targets)
    total_loss = sum(losses.values())
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients exist
    assert outputs["pred_logits"].grad is not None, "pred_logits should have gradient"
    assert outputs["pred_boxes"].grad is not None, "pred_boxes should have gradient"
    
    print("✓ test_criterion_gradient_flow passed")


def test_criterion_empty_targets():
    """Test handling of empty images (no objects)."""
    criterion = DFINECriterion(
        matcher=MockMatcher(),
        num_classes=80,
        weight_dict={"loss_vfl": 1.0, "loss_fgl": 5.0},
        losses=["vfl", "fgl"],
        num_layers=6,
        reg_max=32
    )
    
    batch_size, num_queries = 2, 50
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, 80),
        "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
        "pred_corners": torch.randn(batch_size, num_queries, 4*33),
        "aux_outputs": [
            {
                "pred_logits": torch.randn(batch_size, num_queries, 80),
                "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
                "pred_corners": torch.randn(batch_size, num_queries, 4*33),
            }
        ]
    }
    
    # One image has objects, one is empty
    targets = [
        {"labels": torch.randint(0, 80, (2,)), "boxes": torch.rand(2, 4)},
        {"labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)}
    ]
    
    losses = criterion(outputs, targets)
    
    # Should still produce valid losses
    for value in losses.values():
        assert torch.isfinite(value), f"Loss should be finite even with empty batch"
    
    print("✓ test_criterion_empty_targets passed")


def test_criterion_multi_layer():
    """Test that criterion processes all layers correctly."""
    criterion = DFINECriterion(
        matcher=MockMatcher(),
        num_classes=80,
        weight_dict={"loss_vfl": 1.0, "loss_fgl": 5.0},
        losses=["vfl", "fgl"],
        num_layers=6,
        reg_max=32
    )
    
    batch_size, num_queries, num_aux_layers = 1, 100, 5
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, 80),
        "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
        "pred_corners": torch.randn(batch_size, num_queries, 4*33),
        "aux_outputs": [
            {
                "pred_logits": torch.randn(batch_size, num_queries, 80),
                "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
                "pred_corners": torch.randn(batch_size, num_queries, 4*33),
            }
            for _ in range(num_aux_layers)
        ]
    }
    
    targets = [{"labels": torch.randint(0, 80, (3,)), "boxes": torch.rand(3, 4)}]
    
    losses = criterion(outputs, targets)
    
    # Check we have losses for all layers
    # Final layer (no prefix) + num_aux_layers auxiliary layers
    num_loss_types = 2  # VFL + FGL
    expected_num_keys = num_loss_types * (1 + num_aux_layers)
    
    assert len(losses) == expected_num_keys, \
        f"Expected {expected_num_keys} loss keys, got {len(losses)}"
    
    # Verify aux layer keys exist
    for i in range(num_aux_layers):
        assert f"aux_{i}_loss_vfl" in losses
        assert f"aux_{i}_loss_fgl" in losses
    
    print("✓ test_criterion_multi_layer passed")


if __name__ == "__main__":
    test_criterion_basic()
    test_criterion_output_keys()
    test_criterion_loss_values()
    test_criterion_gradient_flow()
    test_criterion_empty_targets()
    test_criterion_multi_layer()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
