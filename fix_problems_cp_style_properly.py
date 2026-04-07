import os

base_dir = "/home/enn/workspace/project/AI_Judge/example_researches/D-FINE/learning_path"

problems = {
    "04": {
        "name": "Hybrid Encoder Projectors",
        "desc": "Map multi-scale CNN backbone features to a uniform `hidden_dim`.\nIn object detection, the backbone extracts features at different resolutions (scales). To fuse them, they must all be projected to the same number of channels.",
        "in": "`features`: A list of PyTorch tensors `[P3, P4, P5]`.\n- `P3` shape: `(B, 512, H1, W1)`\n- `P4` shape: `(B, 1024, H2, W2)`\n- `P5` shape: `(B, 2048, H3, W3)`",
        "out": "A list of projected tensors `[P3_proj, P4_proj, P5_proj]`.\n- All output tensors must have `C = 256` (the uniform `hidden_dim`).\n- Spatial dimensions `(H, W)` remain unchanged.",
        "const": "- Batch size $B \ge 1$\n- $H_i, W_i > 0$",
        "ex_in": "features = [torch.randn(2, 512, 80, 80), torch.randn(2, 1024, 40, 40), torch.randn(2, 2048, 20, 20)]",
        "ex_out": "[tensor of shape (2, 256, 80, 80), tensor of shape (2, 256, 40, 40), tensor of shape (2, 256, 20, 20)]"
    },
    "05": {
        "name": "BiFPN Feature Fusion",
        "desc": "Implement a simplified Top-Down and Bottom-Up cross-scale interaction (PANet/BiFPN).\nFuses high-level semantic features with low-level spatial features.",
        "in": "`features`: A list of uniform-channel tensors `[P3, P4, P5]`.\n- All have shape `(B, 256, H_i, W_i)` where $H_3 > H_4 > H_5$.",
        "out": "A list of fused tensors `[P3_out, P4_out, P5_out]` of the exact same shapes as the inputs.",
        "const": "- Channel dimension is strictly 256.",
        "ex_in": "features = [torch.randn(1, 256, 80, 80), torch.randn(1, 256, 40, 40), torch.randn(1, 256, 20, 20)]",
        "ex_out": "[tensor (1,256,80,80), tensor (1,256,40,40), tensor (1,256,20,20)]"
    },
    "06": {
        "name": "Contrastive Denoising (CDN) Concept",
        "desc": "Generate noisy anchor boxes and labels from ground truth to simulate imperfect decoder queries. This acts as an auxiliary reconstruction task to accelerate bipartite matching convergence.",
        "in": "`gt_boxes`: Tensor (N, 4) in cxcywh \n`gt_labels`: Tensor (N,) of class IDs in [0, 79]",
        "out": "`noisy_boxes`: Tensor (N, 4) in cxcywh\n`label_emb`: Tensor (N, 256) of class embeddings",
        "const": "- Elements in `gt_labels` $\in [0, \text{num\_classes}-1]$.",
        "ex_in": "gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])\ngt_labels = torch.tensor([5])",
        "ex_out": "noisy_boxes: [[0.51, 0.49, 0.22, 0.18]], label_emb: (1, 256)"
    },
    "07": {
        "name": "D-FINE Positional Encoding",
        "desc": "Compute 2D Sine-Cosine positional embeddings for spatial feature maps to give the Transformer spatial awareness.",
        "in": "`mask`: Boolean tensor of shape `(B, H, W)` indicating valid image regions.",
        "out": "`pos`: Float tensor of shape `(B, 256, H, W)`.",
        "const": "- `hidden_dim` must be even (e.g., 256).\n- `temperature` default is 10000.",
        "ex_in": "mask = torch.ones((1, 10, 10), dtype=torch.bool)",
        "ex_out": "tensor of shape (1, 256, 10, 10)"
    },
    "08": {
        "name": "D-FINE Decoder Cross-Attention",
        "desc": "Implement a standard or deformable multi-head cross-attention layer where object queries attend to the multi-scale feature maps.",
        "in": "`query`: Tensor `(B, NUM_QUERIES, 256)`\n`key`: Tensor `(B, SPATIAL_LEN, 256)`\n`value`: Tensor `(B, SPATIAL_LEN, 256)`",
        "out": "`attn_out`: Tensor `(B, NUM_QUERIES, 256)`",
        "const": "- `NUM_QUERIES` typically 300.\n- `SPATIAL_LEN` is the flattened HxW of feature maps.",
        "ex_in": "query = torch.randn(2, 300, 256), key=value=torch.randn(2, 1000, 256)",
        "ex_out": "tensor of shape (2, 300, 256)"
    },
    "09": {
        "name": "Fine-Grained Box Refinement",
        "desc": "The core D-FINE novelty: maps decoder features to bounding box offsets and applies them to prior reference boxes.",
        "in": "`features`: Tensor `(B, NUM_QUERIES, 256)`\n`reference_boxes`: Tensor `(B, NUM_QUERIES, 4)` in unnormalized logit space or standard space.",
        "out": "`refined_boxes`: Tensor `(B, NUM_QUERIES, 4)` in valid `(0, 1)` range.",
        "const": "- Output boxes must be run through a sigmoid to constrain them to `[0, 1]`.",
        "ex_in": "features = torch.randn(1, 10, 256), ref_boxes = torch.full((1, 10, 4), 0.5)",
        "ex_out": "tensor of shape (1, 10, 4) with values in (0, 1)"
    },
    "10": {
        "name": "Decoupled Decoder Heads",
        "desc": "Separate the refined object query features into a classification branch and a bounding-box regression branch.",
        "in": "`query_features`: Tensor `(B, NUM_QUERIES, 256)`",
        "out": "`cls_logits`: Tensor `(B, NUM_QUERIES, NUM_CLASSES)`\n`box_offsets`: Tensor `(B, NUM_QUERIES, 4)`",
        "const": "- Classification output uses raw logits (no sigmoid).\n- Box output uses raw logits.",
        "ex_in": "query_features = torch.randn(2, 300, 256), NUM_CLASSES=80",
        "ex_out": "cls_logits: (2, 300, 80), box_offsets: (2, 300, 4)"
    },
    "11": {
        "name": "Bipartite Matching Cost Matrix",
        "desc": "Compute the N x M bipartite cost matrix between N predicted queries and M ground truth targets using Focal Loss and L1 distance.",
        "in": "`pred_logits`: `(B, N, NUM_CLASSES)`\n`pred_boxes`: `(B, N, 4)`\n`gt_labels`: `(M,)`\n`gt_boxes`: `(M, 4)`",
        "out": "`cost_matrix`: Tensor `(B, N, M)` representing the cost of assigning query i to target j.",
        "const": "- Probabilities extracted using sigmoid.\n- Target labels must be valid indices.",
        "ex_in": "pred_logits = torch.randn(1, 300, 80), pred_boxes = torch.rand(1, 300, 4)\ngt_labels = torch.tensor([5]), gt_boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])",
        "ex_out": "cost_matrix: (1, 300, 1)"
    },
    "12": {
        "name": "Hungarian Matching Indices",
        "desc": "Solve the global optimum for set assignment using the cost matrix via scipy's `linear_sum_assignment`.",
        "in": "`cost_matrix`: Numpy array or PyTorch tensor `(B, N, M)`.",
        "out": "A list (length B) of tuples `(query_indices, target_indices)`.",
        "const": "- Returns lists of index tensors for indexing into the batches.",
        "ex_in": "C = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]]])",
        "ex_out": "[(tensor([0, 1]), tensor([0, 1]))]"
    },
    "13": {
        "name": "Set Criterion (Main Loss)",
        "desc": "Compute aggregate loss by matching targets against predictions using the Hungarian matching indices.",
        "in": "`outputs`: dict containing `pred_logits` (B, N, NUM_CLASSES) and `pred_boxes` (B, N, 4)\n`targets`: list of dicts with `labels` and `boxes`",
        "out": "A dictionary of scalar loss tensors (e.g., `{'loss_ce': tensor(0.5), 'loss_bbox': tensor(0.2)}`).",
        "const": "- Apply proper reduction (mean over number of matched targets).",
        "ex_in": "outputs = {'pred_logits': torch.randn(1, 300, 80), 'pred_boxes': torch.rand(1, 300, 4)}",
        "ex_out": "dict with scalar float tensors."
    },
    "14": {
        "name": "Set Criterion (Denoising Aux Loss)",
        "desc": "Integrate Contrastive Denoising reconstruction losses. Since we know the index match for denoising queries (they aren't bipartite matched), calculate loss directly.",
        "in": "`dn_outputs`: dict of predictions for denoising queries.\n`dn_targets`: dict of known targets.",
        "out": "A dictionary of scalar loss tensors (e.g., `{'loss_dn_ce': tensor(0.1)}`).",
        "const": "",
        "ex_in": "dn_outputs = {'pred_boxes': torch.rand(1, 20, 4)}",
        "ex_out": "{'loss_dn_bbox': tensor(0.01)}"
    },
    "15": {
        "name": "Post-processing & Top-K Extraction",
        "desc": "Map raw logits and coordinates into final bounding box predictions grouped by top-K confidence.",
        "in": "`outputs`: dict with `pred_logits` and `pred_boxes`\n`target_sizes`: Tensor `(B, 2)` of original image sizes (H, W)",
        "out": "A list of dicts (length B) containing `scores` (K,), `labels` (K,), and `boxes` (K, 4) in absolute pixels.",
        "const": "- Use `torch.topk` on flattened probability scores.",
        "ex_in": "outputs = {'pred_logits': torch.randn(1, 300, 80), 'pred_boxes': torch.rand(1, 300, 4)}",
        "ex_out": "[{'scores': (300,), 'labels': (300,), 'boxes': (300, 4)}]"
    },
    "16": {
        "name": "End-to-End D-FINE Forward Pass",
        "desc": "The complete graph assembly connecting Backbone -> Encoder -> Decoder.",
        "in": "`x`: Image batch tensor `(B, 3, H, W)`",
        "out": "`outputs`: Dict containing all final predictions, auxiliary heads, and denoising outputs if in training mode.",
        "const": "- Check if model `.training` to branch logic.",
        "ex_in": "x = torch.randn(1, 3, 640, 640)",
        "ex_out": "{'pred_logits': (1, 300, 80), 'pred_boxes': (1, 300, 4)}"
    }
}

for id_str, p in problems.items():
    folder = os.path.join(base_dir, f"problem_{id_str}")
    if not os.path.exists(folder): 
        continue
    
    filepath = os.path.join(folder, "problem.md")
    
    new_content = f"""# Problem {id_str} - {p['name']}

## Description
{p['desc']}

## Input Format
{p['in']}

## Output Format
{p['out']}

## Constraints
{p['const']}

## Example
**Input:**
```python
{p['ex_in']}
```

**Output:**
```python
# Expected output signature
{p['ex_out']}
```

## Hints
- Check your broadcasting dimensions meticulously.
- Read the D-FINE paper constraints if shapes mismatch.
- Do not use generic python `for` loops where tensor operations suffice.

## Checker
Run the provided checker to validate your implementation:
`python checker.py`
"""
    with open(filepath, "w") as f:
        f.write(new_content)

print("Rewrote problem.md files correctly with specific details for each!")
