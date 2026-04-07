# Problem 11 - End-to-End D-FINE Model Assembly

## Description
This capstone problem assembles all previously implemented modules into a minimal but complete D-FINE forward pass. The `DFINEMini` model chains: HGNetV2Stem → 2D Positional Encoding → Transformer Encoder → Object Queries → Transformer Decoder → Classification Head + Box Regression Head. Your task is to implement the full `forward()` method that takes a batch of images and returns class logits and predicted bounding boxes.

## Input Format
`images`: `(B, 3, H, W)`, float32 — preprocessed RGB images, `H=W=256` for testing.

## Output Format
dict with:
- `pred_logits`: `(B, N, num_classes)`, float32 — raw class scores for N queries.
- `pred_boxes`:  `(B, N, 4)`, float32 — predicted boxes in normalized `cxcywh`.

## Constraints
- N = `num_queries` (default 100 for this simplified model).
- `num_classes` = 80 (COCO).
- `d_model` = 256.
- num_encoder_layers = 2, num_decoder_layers = 3.
- Box head must output values in `[0, 1]` (apply sigmoid to raw predictions for center/wh).
- All modules must reuse implementations from P06–P10.

## Example
```python
model = DFINEMini(num_classes=80, num_queries=100, d_model=256)
images = torch.randn(2, 3, 256, 256)
out = model(images)
assert out["pred_logits"].shape == (2, 100, 80)
assert out["pred_boxes"].shape  == (2, 100, 4)
assert out["pred_boxes"].min() >= 0 and out["pred_boxes"].max() <= 1
```

## Hints
- Backbone out: apply HGNetV2Stem to get `(B, 96, 32, 32)` (for 256×256 input).
- Project backbone channels to `d_model`: `nn.Conv2d(96, d_model, 1)`.
- Flatten spatial: `(B, d_model, H', W') → (B, H'*W', d_model)`.
- Generate PE: `PositionEmbeddingSine2D(d_model)(x)` where x is backbone feat → `(B, H'*W', d_model)`.
- Encode: apply N encoder layers to (src, pos).
- Learnable query embeddings: `nn.Embedding(num_queries, d_model)`.
- Learnable query pos: `nn.Embedding(num_queries, d_model)`.
- Decode: apply N decoder layers, passing memory=encoder_out, memory_pos=pos.
- Class head: `nn.Linear(d_model, num_classes)`.
- Box head: `nn.Sequential(Linear(d_model, d_model), ReLU, Linear(d_model, 4))`, then sigmoid.

## Checker
```bash
python checker.py
```
