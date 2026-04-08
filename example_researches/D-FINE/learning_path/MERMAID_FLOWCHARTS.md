# D-FINE Architecture: Mermaid Flowcharts

Interactive flowcharts and state diagrams for D-FINE training pipeline.

---

## 1. Overall Training Loop

```mermaid
graph TD
    A["Load Batch<br/>(images, targets)"] --> B["Forward Pass<br/>Model"]
    B --> C["outputs = <br/>pred_logits, pred_boxes,<br/>pred_corners, aux_outputs"]
    C --> D["Criterion Forward<br/>Multi-layer Supervision"]
    D --> E["Internal Matcher<br/>Call 6 times<br/>final + 5 aux layers"]
    E --> F["Compute Matching Union<br/>GO Consensus Indices"]
    F --> G["Compute VFL Loss<br/>Classification"]
    F --> H["Compute FGL Loss<br/>Localization"]
    G --> I["Aggregate Losses<br/>All 12 values"]
    H --> I
    I --> J["total_loss =<br/>sum"]
    J --> K["Backpropagation<br/>backward"]
    K --> L["Optimizer Step<br/>Update all params"]
    L --> M["Next Batch"]
    M -.->|Epoch loop| A
```

---

## 2. DFINECriterion Multi-Layer Flow

```mermaid
graph TD
    A["outputs + targets<br/>input"] --> B["Step 1:<br/>Match Final Layer"]
    B --> C["indices_final"]
    A --> D["Step 2:<br/>Match Aux Layers"]
    D --> E["indices_aux_list<br/>5 sets of indices"]
    C --> F["Step 3:<br/>Compute Matching Union"]
    E --> F
    F --> G["indices_go<br/>Consensus matching"]
    C --> H["Step 4:<br/>Compute Final Layer Loss"]
    E --> I["Step 5:<br/>Loop - Compute Aux Losses"]
    H --> J["loss_vfl, loss_fgl"]
    I --> K["aux_0_loss_vfl, aux_0_loss_fgl<br/>...<br/>aux_4_loss_vfl, aux_4_loss_fgl"]
    J --> L["Aggregate All Losses"]
    K --> L
    L --> M["Return losses dict<br/>12 values total"]
```

---

## 3. Matching Union Algorithm

```mermaid
graph TD
    A["Input:<br/>indices_aux_list<br/>5 layers of matches"] --> B["For each layer:<br/>For each pred→tgt match"]
    B --> C["all_matches<br/>Counter dict"]
    C --> D["aggregates:<br/>pred_id: Counter<br/>tgt_id → vote_count"]
    D --> E["Apply Threshold<br/>keep if votes<br/>> num_layers/2"]
    E --> F["Resolve Conflicts<br/>query with multiple targets"]
    F --> G["Keep highest<br/>vote count target"]
    G --> H["Deduplicate<br/>No target assigned twice"]
    H --> I["indices_go<br/>Consensus indices"]
```

---

## 4. Model Forward: Decoder Layers

```mermaid
graph TD
    A["Encoder Output<br/>(B, L, D)"] --> B["Query Init<br/>(B, N, D)"]
    B --> C["Decoder Layer 0"]
    C --> D0["Pred Head 0<br/>logits_0, boxes_0<br/>corners_0"]
    D0 --> E
    C --> E["Layer 1 Input"]
    E --> F["Decoder Layer 1"]
    F --> D1["Pred Head 1"]
    D1 --> G
    F --> G["Layer 2 Input"]
    G --> H["Decoder Layer 2"]
    H --> D2["Pred Head 2"]
    D2 --> I
    H --> I["Layer 3 Input"]
    I --> J["Decoder Layer 3"]
    J --> D3["Pred Head 3"]
    D3 --> K
    J --> K["Layer 4 Input"]
    K --> L["Decoder Layer 4"]
    L --> D4["Pred Head 4"]
    D4 --> M
    L --> M["Layer 5 Input"]
    M --> N["Decoder Layer 5<br/>FINAL"]
    N --> D5["Pred Head 5<br/>FINAL OUTPUT"]
    D5 --> O["Return all outputs<br/>final + 5 aux"]
    D0 --> AUX["aux_outputs<br/>Store 0-4"]
    D1 --> AUX
    D2 --> AUX
    D3 --> AUX
    D4 --> AUX
```

---

## 5. Matching Process: Single Layer

```mermaid
graph TD
    A["Predictions<br/>(B, N, 4)"] --> B["Target Boxes<br/>(M, 4)"]
    A --> C["Compute Cost Matrix<br/>N×M matrix"]
    B --> C
    C --> D["Class Cost<br/>-log(prob[target_class])"]
    C --> E["L1 Box Cost<br/>L1 distance"]
    C --> F["GIoU Cost<br/>1 - GIoU"]
    D --> G["Combined Cost"]
    E --> G
    F --> G
    G --> H["Hungarian Algorithm<br/>scipy.linear_sum_assignment"]
    H --> I["Optimal Assignment<br/>Minimize total cost"]
    I --> J["indices<br/>pred_idx → target_idx"]
```

---

## 6. VFL Loss Computation

```mermaid
graph TD
    A["Matched Pred Logits<br/>(total_matched, C)"] --> B["Create Target Scores<br/>One-hot encoded"]
    C["Matched Target Labels<br/>(total_matched,)"] --> B
    B --> D["Compute VFL<br/>varifocal_loss"]
    D --> E["Internal:<br/>Sigmoid + BCE"]
    E --> F["Apply Focal Weight<br/>for hard cases"]
    F --> G["Weighted Loss<br/>per prediction"]
    G --> H["Aggregate<br/>mean or sum"]
    H --> I["Weighted by<br/>weight_dict"]
    I --> J["Final VFL Loss<br/>Scalar"]
```

---

## 7. FGL Loss Computation

```mermaid
graph TD
    A["Reference Points<br/>(total, 2)"] --> B["bbox2distance"]
    C["Target Boxes<br/>(total, 4)"] --> B
    B --> D["Compute Edge<br/>Distances"]
    D --> E["Clamp to<br/>0-reg_max"]
    E --> F["Create Soft Labels<br/>via interpolation"]
    F --> G["soft_labels<br/>(total, 4, 33)"]
    H["Pred Corners<br/>(total, 4*33)"] --> I["Reshape to<br/>(total*4, 33)"]
    G --> J["Reshape Soft Labels<br/>(total*4, 33)"]
    I --> K["Unimodal Focal Loss"]
    J --> K
    K --> L["Softmax over bins"]
    L --> M["Compute Focal<br/>Weighting"]
    M --> N["Cross-entropy<br/>with focal weight"]
    N --> O["Weighted loss<br/>per sample"]
    O --> P["Quality Weight<br/>by IoU"]
    P --> Q["Aggregate<br/>mean"]
    Q --> R["Weighted by<br/>weight_dict"]
    R --> S["Final FGL Loss<br/>Scalar"]
```

---

## 8. Gradient Flow Through Layers

```mermaid
graph BT
    A["total_loss<br/>scalar"] --> B["loss_vfl +<br/>loss_fgl"]
    A --> C["aux_0_loss_vfl +<br/>aux_0_loss_fgl"]
    A --> D["aux_4_loss_vfl +<br/>aux_4_loss_fgl"]
    B --> E["Final Decoder<br/>∂loss/∂layer5"]
    C --> F["Aux 0 Decoder<br/>∂loss/∂layer0"]
    D --> G["Aux 4 Decoder<br/>∂loss/∂layer4"]
    E --> H["Encoder<br/>∂loss/∂encoder"]
    F --> H
    G --> H
    H --> I["Backbone<br/>∂loss/∂backbone"]
    I --> J["Update All Parameters!"]
```

---

## 9. Output Dictionary Structure

```mermaid
graph TD
    A["Model Output"] --> B["Final Layer<br/>pred_logits<br/>pred_boxes<br/>pred_corners<br/>ref_points"]
    A --> C["Auxiliary<br/>5 dicts<br/>aux_0...aux_4<br/>each has logits,boxes,corners"]
    A --> D["Metadata<br/>enc_aux_outputs<br/>pre_outputs<br/>reg_scale<br/>up"]
    B --> E["To Criterion"]
    C --> E
    D --> E
    E --> F["Matcher<br/>Indices"]
    E --> G["Loss<br/>Computation"]
    F --> G
    G --> H["12 Loss Values"]
```

---

## 10. Multi-Layer Supervision Philosophy

```mermaid
graph LR
    A["Single-Layer DETR"] --> B["Final Layer Only"]
    B --> C["Predictions from layer 5"]
    C --> D["Only layer 5 gets supervised"]
    D --> E["Shallow gradient flow"]
    E --> F["❌ Early layers<br/>not optimized"]
    
    G["Multi-Layer D-FINE"] --> H["6 Decoders"]
    H --> I["Each layer produces<br/>predictions"]
    I --> J["Each layer supervised<br/>independently"]
    J --> K["Deep gradient flow"]
    K --> L["✓ All layers<br/>optimized"]
    K --> M["✓ Early layers refine<br/>intermediate results"]
    K --> N["✓ Consensus matching<br/>improves robustness"]
```

---

## 11. Data Flow: Batch Processing

```mermaid
graph LR
    A["Batch<br/>B=2 images"] --> B["Backbone"]
    B --> C["Encoder"]
    C --> D["Decoder ×6"]
    D --> E["6 Output Sets"]
    E --> F["Flattened for Loss"]
    F --> G["All 12 losses<br/>aggregated"]
    G --> H["Single total_loss"]
    
    subgraph "Batch Processing"
        I["img_0<br/>3 objects"]
        J["img_1<br/>2 objects"]
    end
    
    A -.-> I
    A -.-> J
    
    K["Irregular batch<br/>Variable object counts"]
    I -.-> K
    J -.-> K
```

---

## 12. Tensor Shape Transformation

```mermaid
graph TD
    A["Images<br/>B, 3, 256, 256<br/>2, 3, 256, 256"] --> B["Backbone"]
    B --> C["Features<br/>Multi-scale<br/>C3, C4, C5"]
    C --> D["Neck<br/>FPN"]
    D --> E["Aligned<br/>4 levels<br/>128 dims"]
    E --> F["Flatten<br/>Encoder input"]
    F --> G["B, L, D<br/>2, 4096, 256"]
    G --> H["Encoder"]
    H --> I["Encoder out<br/>2, 4096, 256"]
    I --> J["Decoder ×6"]
    J --> K["Decoder out<br/>2, 50, 256"]
    K --> L["Pred Heads"]
    L --> M["Logits<br/>2, 50, 80"]
    L --> N["Boxes<br/>2, 50, 4"]
    L --> O["Corners<br/>2, 50, 132"]
```

---

## Key Metrics & Notation

| Symbol | Meaning | Typical Value |
|--------|---------|---|
| **B** | Batch size | 2, 4, 8 |
| **N** | Number of queries | 50, 100 |
| **C** | Number of classes | 80 (COCO) |
| **D** | Model dimension | 256, 512 |
| **L** | Token length (H×W) | 4096 (64×64) |
| **M** | Number of ground truth objects | Variable (0-10+) |
| **reg_max** | FGL distance bins | 32 |
| **num_layers** | Decoder layers | 6 |
| **num_heads** | Attention heads | 8 |

---

These Mermaid diagrams provide interactive, easy-to-understand visualizations of the D-FINE architecture!
