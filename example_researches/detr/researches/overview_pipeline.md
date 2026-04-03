# DETR Pipeline Overview

This document provides a comprehensive technical overview of the **DETR** (Detection Transformer) pipeline, including the architecture, data flow, tensor shapes, and the purpose of each component.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Data Pipeline](#data-pipeline)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Module Reference](#module-reference)

---

## High-Level Architecture

DETR replaces traditional hand-crafted object detection pipelines with an end-to-end Transformer-based approach. The key innovation is treating object detection as a **direct set prediction problem** rather than relying on region proposals (RPN) or non-maximum suppression (NMS).

### Pipeline Overview Diagram

```
┌─────────────────┐
│  Input Image    │  Shape: [B, 3, H, W]
│ (RGB, Padded)   │
└────────┬────────┘
         │
         ↓
    ┌─────────────────────────────────────┐
    │  1. BACKBONE (ResNet50/101)         │  Output: Multi-scale feature maps
    │  - Conv layers with pooling         │  Shape: [B, C, H/32, W/32]
    └────────┬────────────────────────────┘
             │
             ↓
    ┌──────────────────────────────────────┐
    │  2. POSITIONAL ENCODING              │  Encodes spatial information
    │  - Sine or Learned embeddings        │  Shape: [B, 256, H/32, W/32]
    └────────┬─────────────────────────────┘
             │
             ├─────────┬──────────────────────┐
             │         │                      │
             ↓         ↓                      ↓
    ┌──────────────┐ Concat   ┌─────────────────────────────────────┐
    │  Flattened   │ with     │  3. TRANSFORMER ENCODER             │
    │  Features    │ Pos.Emb. │  - 6 encoder layers                 │
    │  [B,HW,256]  │          │  - Self-attention + FFN             │
    │              │          │  Output: Encoded memory             │
    └──────────────┘          │  Shape: [B, HW, 256]                │
                              └────────┬────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ↓                              ↓                              ↓
    ┌─────────────────┐    ┌──────────────────────────────────────┐
    │  Query Embeddings│    │  4. TRANSFORMER DECODER              │
    │  (Learned)      │    │  - 6 decoder layers                  │
    │  [100, 256]     │    │  - Cross-attention to memory         │
    │                 │    │  - Self-attention between queries    │
    │  (100 queries)  │    │  - FFN for each query                │
    └────────┬────────┘    │  Output: Decoded features            │
             │             │  Shape: [B, 100, 256]                │
             │             └────────┬────────────────────────────┘
             │                      │
             └──────────────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ↓                   ↓                   ↓
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Class Head   │  │ Bbox Head    │  │ Mask Head    │
    │ Linear       │  │ (optional)   │  │ (optional)   │
    │ [B,100,K+1]  │  │ MLP+Sigmoid  │  │ Conv layers  │
    │              │  │ [B,100,4]    │  │ [B,100,H,W]  │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                  │
           └─────────────────┼──────────────────┘
                             │
                    ┌────────↓────────┐
                    │  Model Outputs  │
                    │  {pred_logits,  │
                    │   pred_boxes,   │
                    │   pred_masks}   │
                    └────────┬────────┘
                             │
           ┌─────────────────┴──────────────────┐
           │                                    │
        ┌──↓────┐                         ┌──────↓──────┐
        │TRAINING│                        │ INFERENCE   │
        │ (next) │                        │ (next)      │
        └────────┘                        └─────────────┘
```

---

## Data Pipeline

### Input: COCO Detection Dataset

The data pipeline loads images and annotations from the **COCO dataset** and preprocesses them for DETR.

#### Dataset Loading

**File**: `datasets/coco.py`

```python
class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Loads COCO images and annotations
    - Converts polygon masks to binary masks (if return_masks=True)
    - Applies data augmentation transforms
    - Returns (image, target) pairs
    """
```

#### Input Data Format

**Image Tensor**:
- **Shape**: `[B, 3, H_padded, W_padded]`
- **Range**: `[0, 1]` (normalized)
- **Content**: RGB images, padded to the same size within a batch
- **Storage Format**: `NestedTensor` (includes padding mask)

**Padding Mask**:
- **Shape**: `[B, H_padded, W_padded]`
- **Type**: Boolean
- **Meaning**: `True` where padding is present, `False` where image is real
- **Purpose**: Prevents attention from attending to padded regions

#### Ground Truth Target Format

Each target is a dictionary containing:

```python
target = {
    'image_id': torch.tensor([img_id]),           # Shape: [1]
    'orig_size': torch.tensor([H_orig, W_orig]),  # Original image size
    'size': torch.tensor([H_padded, W_padded]),   # Padded image size
    'labels': torch.tensor([...]),                # Shape: [N_gt], class indices
    'boxes': torch.tensor([...]),                 # Shape: [N_gt, 4], normalized (cx, cy, w, h)
    'masks': torch.tensor([...]),                 # Shape: [N_gt, H, W] (optional)
        # Ground truth bounding boxes in format:
        # - cx, cy: center coordinates normalized to [0, 1]
        # - w, h: width and height normalized to [0, 1]
}
```

#### Data Augmentation

**File**: `datasets/transforms.py`

Standard augmentations applied:
- Random horizontal flips
- Color jittering
- Random crops
- Random affine transformations
- Normalization with ImageNet statistics

---

## Model Architecture

### 1. Backbone: Feature Extraction

**File**: `models/backbone.py`

**Purpose**: Extract multi-scale features from the input image using a convolutional neural network.

#### ResNet Backbone (Default: ResNet50)

```
Input: [B, 3, H, W]
       ↓
Conv1 (7x7, stride=2) → [B, 64, H/2, W/2]
       ↓
Layer1 (residual blocks) → [B, 256, H/2, W/2]
       ↓
Layer2 (stride=2) → [B, 512, H/4, W/4]
       ↓
Layer3 (stride=2) → [B, 1024, H/8, W/8]
       ↓
Layer4 (stride=2) → [B, 2048, H/32, W/32]
       ↓
Output: [B, 2048, H/32, W/32]
```

**Intermediate Features** (when `return_interm_layers=True`):
- Layer1 output: `[B, 256, H/2, W/2]`
- Layer2 output: `[B, 512, H/4, W/4]`
- Layer3 output: `[B, 1024, H/8, W/8]`
- Layer4 output: `[B, 2048, H/32, W/32]` (used for transformer)

#### Features Wrapper: NestedTensor

```python
features = NestedTensor(
    tensors=feature_map,    # [B, C, H, W]
    mask=padding_mask        # [B, H, W]
)
```

### 2. Positional Encoding

**File**: `models/position_encoding.py`

**Purpose**: Encode spatial information for the transformer to understand object locations.

#### Two Options:

**a) Sine Positional Encoding** (Default)
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

where:
  pos = position in the sequence
  i = dimension index
  d = d_model = 256
```

**Computation**:
```
Input: [B, H, W] (feature map spatial dimensions)

For each spatial position (y, x):
  1. Generate sinusoidal embeddings for x coordinate
  2. Generate sinusoidal embeddings for y coordinate
  3. Concatenate: [sin(x), cos(x), sin(y), cos(y), ...]
  
Output: [B, 256, H, W]
```

**b) Learned Positional Encoding**
```
- Row embedding: nn.Embedding(H_max, d_model/2)
- Col embedding: nn.Embedding(W_max, d_model/2)
- Concatenate and tile to feature map size
```

**Output Shape**: `[B, 256, H/32, W/32]`

### 3. Flattening and Projection

**Purpose**: Convert 2D feature maps into sequence format for the transformer.

```
Input: [B, 2048, H/32, W/32]

Step 1: Project to hidden dimension
  Conv2d(2048 → 256, kernel_size=1)
  Output: [B, 256, H/32, W/32]

Step 2: Flatten spatial dimensions
  [B, 256, H/32, W/32] → [B, 256, HW] → [HW, B, 256]
  (Transformer uses sequence-first format)

where HW = (H/32) × (W/32) ≈ 1024 for a 1024×1024 image

Output: [HW, B, 256]  # Sequence length ~ 1024
```

### 4. Transformer Encoder

**File**: `models/transformer.py`

**Purpose**: Encode spatial features and capture global context through multi-head self-attention.

#### Architecture

```
Input: 
  - src: [HW, B, 256]        # Flattened features
  - mask: [B, HW]            # Padding mask
  - pos_embed: [HW, B, 256]  # Positional encodings

N = 6 encoder layers (configurable via --enc_layers)

For each layer:
  1. Multi-head self-attention
     - Queries, Keys, Values from src
     - Attention mask applied to prevent attending to padding
     - Positional encoding added to queries and keys
     - Number of heads: 8 (--nheads)
  
  2. Feed-forward network
     - Linear(256 → 2048)
     - ReLU
     - Linear(2048 → 256)
  
  3. Residual connections + Layer normalization

Output: [HW, B, 256]  # Encoded memory
```

**Key Differences from Standard Transformer**:
- Positional encodings are added to queries and keys in attention, not at the input
- Padding mask prevents attention to padded positions
- No activation function between self-attention and FFN (pre-norm architecture)

### 5. Transformer Decoder

**File**: `models/transformer.py`

**Purpose**: Predict object detections through iterative refinement using learned query embeddings.

#### Query Embeddings

```python
self.query_embed = nn.Embedding(num_queries=100, d_model=256)
# 100 learnable query embeddings
# Each query learns to "specialize" in detecting certain types of objects
# Shape: [100, 256]
```

#### Decoder Forward Pass

```
Input:
  - tgt: [100, B, 256]       # Query embeddings, initialized to zeros initially
  - memory: [HW, B, 256]     # Encoder output
  - query_pos: [100, B, 256] # Query embeddings (used as positional encoding)
  - pos_embed: [HW, B, 256]  # Encoder positional encodings

N = 6 decoder layers

For each layer:
  1. Self-attention on queries
     - Queries, Keys, Values from tgt (queries attend to other queries)
     - Query positions added
     - No attention mask
  
  2. Cross-attention
     - Queries from tgt
     - Keys and Values from memory (encoder output)
     - Decoder queries attend to encoder features
     - Positional encodings added to encoder features (keys)
  
  3. Feed-forward network
     - Linear(256 → 2048)
     - ReLU
     - Linear(2048 → 256)

Output:
  - hs: [6, B, 100, 256]  # Outputs from all 6 decoder layers
        Each layer produces predictions that are refined by subsequent layers
  - memory: [HW, B, 256]  # Encoder output (used for auxiliary losses)
```

**Why This Works**:
- Query embeddings act as "object templates" that learn what features to look for
- Cross-attention allows queries to directly attend to image features
- Self-attention between queries allows queries to avoid predicting duplicate objects
- Progressive refinement through 6 layers improves prediction quality

### 6. Detection Heads

**File**: `models/detr.py`

#### Classification Head

```python
self.class_embed = nn.Linear(256, num_classes + 1)

Input: [B, 100, 256]  (decoder output)
Output: [B, 100, num_classes + 1]
        81 classes for COCO (80 object classes + 1 background class)

Each of 100 queries predicts a class logit for each possible class.
The (+1) accounts for the "no object" class.
```

#### Bounding Box Regression Head

```python
self.bbox_embed = MLP(256, 256, 4, 3)  # 3-layer MLP
# Layer 1: Linear(256 → 256)
# Layer 2: Linear(256 → 256)
# Layer 3: Linear(256 → 4)

Input: [B, 100, 256]  (decoder output)
Output: [B, 100, 4]   (after sigmoid)
        
Box format: (center_x, center_y, width, height)
All values normalized to [0, 1] after sigmoid.
```

#### Instance Segmentation Head (Optional)

**File**: `models/segmentation.py`

```
Used only if --masks flag is set

1. Bbox Attention: Attention map from query→memory
   Output: [B, num_heads, HW]
   
2. Mask Head: Convolutional network using FPN
   Input: Multi-scale encoder features + attention map
   Output: [B, 100, H, W]  # Binary mask for each query
```

---

## Training Pipeline

### Forward Pass

```
Input: batch of images and targets

1. Feature Extraction (Backbone)
   images: [B, 3, H, W] → features: [B, 256, H/32, W/32]

2. Positional Encoding
   pos: [B, 256, H/32, W/32]

3. Transformer
   features + pos → transformed features: [B, 100, 256]

4. Detection Heads
   features → logits: [B, 100, 81]
   features → boxes: [B, 100, 4]

Output dict:
{
    'pred_logits': [B, 100, 81],     # Raw classification logits
    'pred_boxes': [B, 100, 4],       # Normalized box coordinates
    'aux_outputs': [                 # Auxiliary losses (6 intermediate layers)
        {'pred_logits': [...], 'pred_boxes': [...]},
        ...
    ]
}
```

### Matching: Hungarian Algorithm

**File**: `models/matcher.py`

**Purpose**: Assign ground truth objects to predicted objects in an optimal way.

#### Matching Cost Computation

```
For each predicted query and ground truth object, compute a cost:

cost = α·cost_class + β·cost_bbox + γ·cost_giou

where:
  α = --set_cost_class = 1 (default)
  β = --set_cost_bbox = 5 (default)
  γ = --set_cost_giou = 2 (default)

cost_class = -probability[ground_truth_class]
             (negative because we're minimizing cost)

cost_bbox = L1 distance between predicted and GT box

cost_giou = 1 - GIoU(predicted_box, GT_box)

Cost matrix shape: [num_predictions * B, num_ground_truths * B]
                   [B * 100, avg(N_gt) * B]
```

#### Hungarian Matching

```
Use scipy.optimize.linear_sum_assignment to find optimal assignment:
- Minimizes total matching cost
- Each prediction matched to at most one ground truth
- Each ground truth matched to at most one prediction
- Unmatched predictions (more than GT) → treated as background
- Unmatched ground truths → ignored (shouldn't happen with enough queries)

Output: List of tuples (pred_indices, gt_indices) for each batch
        pred_indices: which of 100 queries are matched
        gt_indices: which ground truths they matched to
```

### Loss Computation

**File**: `models/detr.py`, class `SetCriterion`

#### Classification Loss (Cross-Entropy)

```
For matched predictions:
  L_class = CE(pred_class, gt_class) with class weights

For unmatched predictions:
  L_class = CE(pred_class, background_class)

Loss is weighted with eos_coef (--eos_coef = 0.1) for background class
to handle class imbalance (only ~50 objects per image, but 100 queries).

Mathematical form:
  L_ce = -Σ w_c * log(P(c)) where c is true class, w_c is class weight
```

#### Bounding Box Loss (L1 + GIoU)

```
Only applied to matched predictions:

L1 loss:
  L_bbox = (1/N) * Σ ||pred_box - gt_box||_1
  
  N = total number of ground truth boxes across batch

GIoU loss:
  L_giou = (1/N) * Σ (1 - GIoU(pred_box, gt_box))
  
  GIoU = IoU - (C \ (A ∪ B)) / C
         where C is smallest enclosing box

Total box loss:
  L_box_total = λ_bbox * L_bbox + λ_giou * L_giou
  λ_bbox = --bbox_loss_coef = 5
  λ_giou = --giou_loss_coef = 2
```

#### Cardinality Loss (Auxiliary, no gradient)

```
@torch.no_grad()
Measures the difference between predicted and actual number of objects.
Used for logging only: does not propagate gradients.
```

#### Segmentation Loss (Optional, Focal + Dice)

```
If --masks flag is set:

L_focal = Focal(pred_mask, gt_mask)
L_dice = 1 - (2 * TP) / (2 * TP + FP + FN)

Both applied only to matched predictions.
```

#### Auxiliary Losses

```
The main loss is computed for all 6 decoder layers:
  1. Loss from layer 1 predictions
  2. Loss from layer 2 predictions
  ...
  6. Loss from layer 6 predictions (the actual predictions)

Each intermediate layer loss has weight 1.0
Total loss = Σ L_layer_i where i ∈ [1, 6]

This provides supervision signals deeper in the network,
improving training convergence and prediction quality.

Note: Mask losses are only computed for the final layer (too expensive).
```

#### Total Loss

```
L_total = Σ_layer L_ce + λ_bbox * L_bbox + λ_giou * L_giou + 
          λ_mask * L_mask + λ_dice * L_dice

Weights in loss computation:
  weight_dict = {
      'loss_ce': 1,
      'loss_bbox': 5,
      'loss_giou': 2,
      'loss_mask': 1,
      'loss_dice': 1
  }
```

### Optimization

**File**: `main.py`, `engine.py`

```
Optimizer: AdamW
Learning rate schedule: polynomial decay with warmup

Hyperparameters:
  - lr = 1e-4            # Learning rate for backbone
  - lr_backbone = 1e-5   # Lower learning rate for backbone (frozen more)
  - weight_decay = 1e-4
  - batch_size = 2 (per GPU)
  - epochs = 300
  - lr_drop = 200        # Drop LR by 0.1 after 200 epochs
  - clip_max_norm = 0.1  # Gradient clipping

Distributed training: DataParallel or DistributedDataParallel (DDP)
```

---

## Inference Pipeline

### Forward Pass

```
Input: Single image (or batch during beamforming)
  [B, 3, H, W]

Through DETR model:
  → backbone features: [B, 2048, H/32, W/32]
  → positional encoding: [B, 256, H/32, W/32]
  → transformer: [B, 100, 256]
  → detection heads: predictions

Output:
{
    'pred_logits': [B, 100, 81],   # Unnormalized scores
    'pred_boxes': [B, 100, 4],     # Normalized box coordinates (cx, cy, w, h)
}
```

### Post-Processing

**File**: `models/detr.py`, class `PostProcess`

#### Class Probability Extraction

```python
# Convert logits to probabilities
probs = pred_logits.softmax(dim=-1)  # [B, 100, 81]

# Extract score and class for each query
scores, classes = probs.max(dim=-1)  # Each: [B, 100]
```

#### Keep High-Confidence Predictions

```python
# Threshold on confidence
threshold = 0.5  # or configurable

keep = scores > threshold

# Filter predictions
pred_boxes = pred_boxes[keep]
scores = scores[keep]
classes = classes[keep]
```

#### Coordinate Denormalization

```
Input (normalized): cx, cy, w, h ∈ [0, 1]
Output (pixel coordinates):

x_min = (cx - w/2) * img_width
y_min = (cy - h/2) * img_height
x_max = (cx + w/2) * img_width
y_max = (cy + h/2) * img_height

Result format: (x_min, y_min, x_max, y_max)
```

#### Instance Segmentation Post-Processing (Optional)

```
If using instance segmentation (--masks):

1. Upsample mask predictions to original image size
2. Apply threshold to convert to binary masks
3. Store masks along with boxes

Output:
{
    'scores': [...],
    'labels': [...],
    'boxes': [...],           # (x_min, y_min, x_max, y_max)
    'masks': [...]            # Binary panoptic masks (if enabled)
}
```

---

## Module Reference

### Key Files and Their Responsibilities

| File | Purpose | Key Components |
|------|---------|-----------------|
| `main.py` | Entry point, argument parsing, training loop orchestration | get_args_parser(), main() |
| `engine.py` | Training and evaluation functions | train_one_epoch(), evaluate() |
| `models/detr.py` | DETR model architecture and loss computation | DETR, SetCriterion, PostProcess |
| `models/backbone.py` | Feature extraction with ResNet | BackboneBase, Backbone, build_backbone() |
| `models/transformer.py` | Transformer encoder and decoder | Transformer, TransformerEncoder, TransformerDecoder |
| `models/position_encoding.py` | Positional encoding strategies | PositionEmbeddingSine, PositionEmbeddingLearned |
| `models/matcher.py` | Hungarian matching algorithm | HungarianMatcher |
| `models/segmentation.py` | Instance segmentation head | DETRsegm, MaskHeadSmallConv |
| `datasets/coco.py` | COCO dataset loading | CocoDetection, ConvertCocoPolysToMask |
| `datasets/transforms.py` | Data augmentation | Compose, RandomHorizontalFlip, ColorJitter, etc. |
| `util/box_ops.py` | Box operations | box_cxcywh_to_xyxy(), generalized_box_iou() |
| `util/misc.py` | Utilities | NestedTensor, MetricLogger, distributed training helpers |

### Configuration Parameters

#### Model Parameters

```
--hidden_dim 256         # Transformer embedding dimension
--nheads 8               # Number of attention heads
--enc_layers 6           # Encoder layers
--dec_layers 6           # Decoder layers
--dim_feedforward 2048   # FFN hidden dimension
--num_queries 100        # Number of detection queries
--backbone resnet50      # Backbone architecture
--position_embedding sine  # Positional encoding type
```

#### Matching and Loss Parameters

```
--set_cost_class 1       # Classification weight in matching cost
--set_cost_bbox 5        # L1 box weight in matching cost
--set_cost_giou 2        # GIoU weight in matching cost

--bbox_loss_coef 5       # L1 loss weight
--giou_loss_coef 2       # GIoU loss weight
--eos_coef 0.1          # Background class weight
```

#### Training Parameters

```
--lr 1e-4               # Learning rate
--lr_backbone 1e-5      # Backbone learning rate
--weight_decay 1e-4     # L2 regularization
--batch_size 2          # Batch size per GPU
--epochs 300            # Training epochs
--lr_drop 200           # Learning rate drop schedule
--clip_max_norm 0.1     # Gradient clipping
--num_workers 2         # Data loading workers
```

---

## Tensor Shape Flow Summary

```
Input Image:              [B, 3, H_orig, W_orig]
After Padding:            [B, 3, H, W]                    H = W = multiple of 32
Padding Mask:             [B, H, W]
                          
Backbone Output:          [B, 2048, H/32, W/32]
Pos Encoding:             [B, 256, H/32, W/32]
                          
Projected Features:       [B, 256, H/32, W/32]
Flattened for TX:         [HW, B, 256]                    HW ≈ 1024
                          
Encoder Memory:           [HW, B, 256]
Decoder Queries:          [100, B, 256]
                          
Decoder Output Stack:     [6, B, 100, 256]                6 layers
Final Output:             [B, 100, 256]
                          
Classification Logits:    [B, 100, 81]                    81 = 80 classes + 1 background
Bounding Boxes:           [B, 100, 4]
Instance Masks (opt):     [B, 100, H, W]

Ground Truth:
  - Labels:               [B, N_gt]                       N_gt varies per image
  - Boxes:                [B, N_gt, 4]
  - Masks (opt):          [B, N_gt, H, W]
```

---

## Summary

DETR represents a paradigm shift in object detection:
- **End-to-end**: No RPN, NMS, or hand-crafted pipeline components
- **Query-based**: 100 learned queries "compete" to predict objects
- **Global context**: Transformer encoder captures full image context
- **Set prediction**: Hungarian matching provides optimal object assignments
- **Extensible**: Mask head can be added for instance segmentation

The key innovation is formulating detection as a **sequence-to-sequence prediction problem**, where the model directly outputs a fixed number of predictions that are matched to ground truth objects using optimal assignment algorithms.
