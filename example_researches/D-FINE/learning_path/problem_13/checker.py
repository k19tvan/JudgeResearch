"""Problem 13: Checker - Validate Matching Union Implementation"""
import torch
from solution import HungarianMatcher


def test_basic_consensus():
    \"\"\"Test basic consensus voting across layers.\"\"\"
    matcher = HungarianMatcher()
    
    # Simulate matching from 3 layers, batch_size=1
    # Layer 0: matches (2,0), (5,1), (7,2)
    # Layer 1: matches (2,0), (5,1), (9,3)  <- (2,0) and (5,1) are consensus
    # Layer 2: matches (2,0), (5,1), (7,2)  <- (2,0) and (5,1) are consensus again
    
    indices_layer_0 = (
        torch.tensor([[2, 5, 7]], dtype=torch.int64),
        torch.tensor([[0, 1, 2]], dtype=torch.int64)
    )
    indices_layer_1 = (
        torch.tensor([[2, 5, 9]], dtype=torch.int64),
        torch.tensor([[0, 1, 3]], dtype=torch.int64)
    )
    indices_layer_2 = (
        torch.tensor([[2, 5, 7]], dtype=torch.int64),
        torch.tensor([[0, 1, 2]], dtype=torch.int64)
    )
    
    indices_list = [indices_layer_0, indices_layer_1, indices_layer_2]
    
    # Compute union
    union = matcher.compute_matching_union(indices_list)
    
    # Check results
    assert len(union) == 1, "Should return list with 1 element (batch_size=1)"
    src, tgt = union[0]
    
    # (2,0) appears 3 times, (5,1) appears 3 times → consensus
    # (7,2) appears 2 times → consensus
    # (9,3) appears 1 time → not in final if threshold=2
    
    # Check that consensus pairs are present
    for i in range(len(src)):
        assert src[i] == 2 or src[i] == 5 or src[i] == 7, f"Unexpected src value: {src[i]}"
        assert tgt[i] == 0 or tgt[i] == 1 or tgt[i] == 2, f"Unexpected tgt value: {tgt[i]}"
    
    print("✓ test_basic_consensus passed")


def test_one_to_one_constraint():
    \"\"\"Test that one-to-one constraint is enforced.\"\"\"
    matcher = HungarianMatcher()
    
    # Conflicting matches where one query maps to multiple targets
    indices_layer_0 = (
        torch.tensor([[2, 2, 5]], dtype=torch.int64),  # srcNote: (2,?) appears twice
        torch.tensor([[0, 1, 2]], dtype=torch.int64)
    )
    indices_layer_1 = (
        torch.tensor([[2, 5]], dtype=torch.int64),
        torch.tensor([[1, 2]], dtype=torch.int64)
    )
    
    indices_list = [indices_layer_0, indices_layer_1]
    union = matcher.compute_matching_union(indices_list)
    
    src, tgt = union[0]
    
    # Check one-to-one: no query appears twice, no target appears twice
    assert len(src) == len(set(src.tolist())), "Query (src) should appear at most once"
    assert len(tgt) == len(set(tgt.tolist())), "Target (tgt) should appear at most once"
    
    print("✓ test_one_to_one_constraint passed")


def test_empty_layer():
    \"\"\"Test handling of empty matching from some layers.\"\"\"
    matcher = HungarianMatcher()
    
    indices_layer_0 = (
        torch.tensor([[2, 5]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64)
    )
    indices_layer_1 = (
        torch.tensor([], dtype=torch.int64),  # Empty layer
        torch.tensor([], dtype=torch.int64)
    )
    indices_layer_2 = (
        torch.tensor([[2, 5]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64)
    )
    
    indices_list = [indices_layer_0, indices_layer_1, indices_layer_2]
    union = matcher.compute_matching_union(indices_list)
    
    src, tgt = union[0]
    
    # Should still produce output from non-empty layers
    assert len(src) > 0, "Should produce matches even with empty layers"
    
    print("✓ test_empty_layer passed")


def test_complete_disagreement():
    \"\"\"Test behavior when all layers disagree (no consensus).\"\"\"
    matcher = HungarianMatcher()
    
    # All layers match different query-target pairs
    indices_layer_0 = (
        torch.tensor([[1]], dtype=torch.int64),
        torch.tensor([[0]], dtype=torch.int64)
    )
    indices_layer_1 = (
        torch.tensor([[2]], dtype=torch.int64),
        torch.tensor([[1]], dtype=torch.int64)
    )
    indices_layer_2 = (
        torch.tensor([[3]], dtype=torch.int64),
        torch.tensor([[2]], dtype=torch.int64)
    )
    
    indices_list = [indices_layer_0, indices_layer_1, indices_layer_2]
    union = matcher.compute_matching_union(indices_list)
    
    src, tgt = union[0]
    
    # When no consensus, should fall back to including all matches
    # But maintain one-to-one
    assert len(src) > 0, "Should produce output even with complete disagreement"
    assert len(src) == len(set(src.tolist())), "One-to-one constraint on src"
    assert len(tgt) == len(set(tgt.tolist())), "One-to-one constraint on tgt"
    
    print("✓ test_complete_disagreement passed")


def test_batch_handling():
    \"\"\"Test handling multiple batch items.\"\"\"
    matcher = HungarianMatcher()
    
    # Batch size = 2
    indices_layer_0 = (
        torch.tensor([[2, 5], [1, 3]], dtype=torch.int64),
        torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    )
    indices_layer_1 = (
        torch.tensor([[2, 5], [1, 4]], dtype=torch.int64),
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    )
    
    indices_list = [indices_layer_0, indices_layer_1]
    union = matcher.compute_matching_union(indices_list)
    
    # Should return list with 2 elements (one per batch item)
    assert len(union) == 2, f"Expected 2 batch items, got {len(union)}"
    
    for src, tgt in union:
        assert isinstance(src, torch.Tensor), "Should return tensors"
        assert isinstance(tgt, torch.Tensor), "Should return tensors"
        assert len(src) == len(tgt), "src and tgt should have same length"
    
    print("✓ test_batch_handling passed")


def test_output_format():
    \"\"\"Test that output format matches specification.\"\"\"
    matcher = HungarianMatcher()
    
    indices_layer_0 = (
        torch.tensor([[2, 5]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64)
    )
    indices_layer_1 = (
        torch.tensor([[2, 5]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64)
    )
    
    indices_list = [indices_layer_0, indices_layer_1]
    union = matcher.compute_matching_union(indices_list)
    
    # Check format
    assert isinstance(union, list), "Output should be a list"
    assert all(isinstance(item, tuple) and len(item) == 2 for item in union), \
        "Each item should be (src, tgt) tuple"
    assert all(isinstance(src, torch.Tensor) and isinstance(tgt, torch.Tensor) \
        for src, tgt in union), "Elements should be tensors"
    assert all(src.dtype == torch.int64 and tgt.dtype == torch.int64 \
        for src, tgt in union), "Tensors should be int64"
    
    print("✓ test_output_format passed")


if __name__ == "__main__":
    test_basic_consensus()
    test_one_to_one_constraint()
    test_empty_layer()
    test_complete_disagreement()
    test_batch_handling()
    test_output_format()
    
    print("\\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
