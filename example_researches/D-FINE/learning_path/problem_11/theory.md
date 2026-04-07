# Problem 11 Theory - End-to-End D-FINE Model Assembly

## Core Definitions
- **End-to-End Object Detector**: Takes raw images → directly outputs bounding boxes + class scores without any handcrafted anchors, NMS, or region proposal stages.
- **Data Flow**: Image → CNN Backbone → Feature Projection → Sine PE → Transformer Encoder → Cross-Attended Decoder (with Object Queries) → Head → Loss.
- **Object Queries**: N learnable embedding vectors (not image-derived), each representing one "slot" that the decoder fills with an object prediction. The decoder learns what object each slot specializes on via training.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `images` | `(B, 3, H, W)` | Input batch |
| `feat` | `(B, 96, H/8, W/8)` | HGNetV2 features |
| `feat_proj` | `(B, d_model, H', W')` | After 1×1 projection |
| `src` | `(B, H'*W', d_model)` | Flattened feature tokens |
| `pos` | `(B, H'*W', d_model)` | Sine 2D PE |
| `memory` | `(B, H'*W', d_model)` | Encoder output |
| `tgt` | `(B, N, d_model)` | Object query embeddings |
| `qpos` | `(B, N, d_model)` | Query positional anchors |
| `dec_out` | `(B, N, d_model)` | Decoder output |
| `pred_logits` | `(B, N, C)` | Class scores |
| `pred_boxes` | `(B, N, 4)` | Boxes in cxcywh [0,1] |

## Main Equations (LaTeX)

**Backbone Feature Flow:**
$$ F = \text{LCS}(\text{Stem}(I)) \in \mathbb{R}^{B \times 96 \times \frac{H}{8} \times \frac{W}{8}} $$

**PE injection:**
$$ \text{src} = \text{flatten}(\text{Conv}_{1\times1}(F)), \quad \text{pos} = \text{SinePE}(\text{src}) $$

**Encoder:**
$$ M = \text{Enc}_L(\ldots\text{Enc}_1(\text{src}, \text{pos})\ldots) $$

**Decoder:**
$$ D = \text{Dec}_L(\ldots\text{Dec}_1(\text{tgt}, M, \text{qpos}, \text{pos})\ldots) $$

**Heads:**
$$ \hat{y}_{\text{cls}} = D W_{cls},\quad \hat{b} = \sigma\!\left(D W_{b2}^T \cdot \text{ReLU}(D W_{b1}^T)\right) $$

## Step-by-Step Computation Flow
1. `_, feat = backbone(images)` → `(B, 96, H', W')`.
2. `feat_p = proj(feat)` → `(B, d, H', W')`.
3. `src = feat_p.flatten(2).transpose(1,2)` → `(B, H'*W', d)`.
4. `pos = pe(feat_p)` → `(B, H'*W', d)`.
5. `memory = src` → apply each encoder layer: `memory = enc_layer(memory, pos)`.
6. `tgt = query_embed.weight.unsqueeze(0).expand(B,-1,-1)`.
7. `qpos = query_pos.weight.unsqueeze(0).expand(B,-1,-1)`.
8. Apply each decoder layer: `tgt = dec_layer(tgt, memory, qpos, pos)`.
9. `pred_logits = cls_head(tgt)`. `pred_boxes = sigmoid(box_head(tgt))`.
10. Return `{"pred_logits": pred_logits, "pred_boxes": pred_boxes}`.

## Tensor Shape Flow
```
images: (B, 3, H, W)
  ↓ HGNetV2Stem
feat: (B, 96, H/8, W/8)
  ↓ Conv 1×1 + flatten
src, pos: (B, T, d_model)  where T = H/8 * W/8
  ↓ Encoder ×2
memory: (B, T, d_model)
                        tgt, qpos: (B, N, d_model)
  ↓ Decoder ×3 (memory + tgt)
dec_out: (B, N, d_model)
  ↓ cls_head           ↓ box_head + sigmoid
pred_logits: (B,N,C)   pred_boxes: (B,N,4) ∈ [0,1]
```

## Practical Interpretation
The full model has only ~8M parameters in this simplified form (vs. the full D-FINE-S at 31M) but demonstrates the entire computational graph. The object queries compete with each other through self-attention to "claim" different objects, while cross-attention fetches the per-object image evidence. This produces N diverse predictions from which the top-scoring ones are selected at inference time (threshold on class probability).
