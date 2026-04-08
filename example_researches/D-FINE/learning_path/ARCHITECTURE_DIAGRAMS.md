# D-FINE Architecture Diagrams & Flowcharts

Complete visual reference for the multi-layer D-FINE architecture with matching consensus and fine-grained localization.

---

## 1. Overall D-FINE Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           D-FINE Training Pipeline                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              Input Images
                                   │
                                   ↓
                        ┌──────────────────────┐
                        │     Backbone         │
                        │  (HGNetV2-S/L/M/N)   │
                        │                      │
                        │  Extract Features    │
                        │  (Multi-scale)       │
                        └──────────┬───────────┘
                                   │
                                   ↓
                        ┌──────────────────────┐
                        │   Multi-Level Neck   │
                        │   (FPN-style)        │
                        │                      │
                        │  C3, C4, C5, C6      │
                        │  (4 feature levels)  │
                        └──────────┬───────────┘
                                   │
                                   ↓
                        ┌──────────────────────┐
                        │   Transformer        │
                        │   Encoder            │
                        │                      │
                        │   Self-attention     │
                        │   over features      │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ↓                             ↓
        ┌─────────────────────┐      ┌─────────────────────┐
        │ Decoder Layer 0     │      │ Reference Points    │
        │ Query→Visual Props  │      │ (Object centers)    │
        └──────────┬──────────┘      └─────────────────────┘
                    │
        ┌───────────┴──────────────────┐
        │                              │
        ↓                              ↓
   Decoder       ┌──────────────────────────────┐
   Layers 1-5    │  Output Prediction Heads     │
   (Iterative)   │                              │
                 │  • pred_logits (B, N, C)     │
                 │  • pred_boxes (B, N, 4)      │
                 │  • pred_corners (B, N, 4*33) │
                 │  • ref_points (B, N, 2)      │
                 └──────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ↓                               ↓
        ┌──────────────────┐         ┌──────────────────────┐
        │  auxiliary_0     │         │  Final Predictions   │
        │  (Layer 0 out)   │         │  (Layer 5 out)       │
        │                  │         │                      │
        │  • pred_logits   │         │  • pred_logits       │
        │  • pred_boxes    │         │  • pred_boxes        │
        │  • pred_corners  │         │  • pred_corners      │
        └────────┬─────────┘         │  • aux_outputs       │
                 │                   │  • enc_aux_outputs   │
                 │                   │  • pre_outputs       │
              ┌──┴────┐              │  • reg_scale         │
              │  ...  │              │  • up                │
              │  ...  │              └──────┬───────────────┘
              └──┬────┘                     │
                 │         ┌────────────────┴────────────────┐
                 │         │                                 │
                 └────────→ All Outputs Sent to Criterion   │
                           │    (Multi-layer supervision)   │
                           └──────────────┬──────────────────┘
                                         │
                                         ↓
                            ┌────────────────────────┐
                            │   DFINECriterion       │
                            │                        │
                            │  1. Match each layer   │
                            │  2. Compute union      │
                            │  3. Aggregate losses   │
                            └────────────┬───────────┘
                                         │
                      ┌──────────────────┼──────────────────┐
                      ↓                  ↓                  ↓
                 loss_vfl           loss_fgl           aux_*_loss_*
                 (Classification)  (Localization)      (All layers)
                      │                  │                  │
                      └──────────────────┬──────────────────┘
                                        │
                                        ↓
                            ┌──────────────────────┐
                            │  total_loss =        │
                            │  sum(losses.values())│
                            └──────────┬───────────┘
                                       │
                                       ↓
                            ┌──────────────────────┐
                            │  Backpropagation     │
                            │  total_loss.backward()
                            └──────────┬───────────┘
                                       │
                                       ↓
                            ┌──────────────────────┐
                            │  Optimizer Step      │
                            │  optimizer.step()    │
                            └──────────────────────┘
```

---

## 2. Multi-Layer Decoder Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer Decoder                          │
│              (6 Decoders Stacked, Each Produces Output)         │
└─────────────────────────────────────────────────────────────────┘

                  Encoder Output (from backbone+neck)
                      │
                      ├→ decoder_layer_0
                      │       │
                      │       ├→ attn_output_0
                      │       ├→ ffn_output_0
                      │       │
                      │       └→ pred_head_0
                      │           │
                      │           ├→ logits_0 (B, N, C)
                      │           ├→ boxes_0 (B, N, 4)
                      │           └→ corners_0 (B, N, 4*33)
                      │
                      ├→ decoder_layer_1
                      │       │
                      │       ├→ ...(same as above)
                      │       └→ pred_head_1
                      │
                      ├→ decoder_layer_2
                      │       └→ ...
                      │
                      ├→ decoder_layer_3
                      │       └→ ...
                      │
                      ├→ decoder_layer_4
                      │       └→ ...
                      │
                      └→ decoder_layer_5 (FINAL)
                              │
                              ├→ attn_output_5
                              ├→ ffn_output_5
                              │
                              └→ pred_head_5
                                  │
                                  ├→ logits_5 (B, N, C)   ← FINAL logits
                                  ├→ boxes_5 (B, N, 4)    ← FINAL boxes
                                  └→ corners_5 (B, N, 4*33) ← FINAL corners

┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT STRUCTURE                           │
│                                                                 │
│  outputs = {                                                    │
│      "pred_logits": logits_5,      # Final layer output       │
│      "pred_boxes": boxes_5,        # Final layer output       │
│      "pred_corners": corners_5,    # Final layer output       │
│      "ref_points": centers_5,                                 │
│      "aux_outputs": [              # Auxiliary from layers 0-4│
│          {logits_0, boxes_0, corners_0},                      │
│          {logits_1, boxes_1, corners_1},                      │
│          {logits_2, boxes_2, corners_2},                      │
│          {logits_3, boxes_3, corners_3},                      │
│          {logits_4, boxes_4, corners_4},                      │
│      ]                                                         │
│  }                                                              │
│                                                                 │
│  Total Outputs: 6 sets of predictions (5 aux + 1 final)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Multi-Layer Supervision Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                    DFINECriterion.forward()                          │
│              Multi-Layer Loss Computation & Aggregation              │
└──────────────────────────────────────────────────────────────────────┘

                        outputs + targets
                              │
                ┌─────────────┴────────────┐
                │                          │
        ┌───────▼──────────┐     ┌────────▼─────────┐
        │  outputs dict    │     │  targets list    │
        │  (6 layer set)   │     │  (batch_size)    │
        │                  │     │                  │
        │ • pred_logits    │     │ • labels (M,)    │
        │ • pred_boxes     │     │ • boxes (M, 4)   │
        │ • pred_corners   │     │ • image_id       │
        │ • aux_outputs[5] │     │                  │
        └────────┬─────────┘     └──────────────────┘
                 │
                 ├═════════════════════════════════════════════════════┐
                 │  STEP 1: MATCH FINAL LAYER                         │
                 ├═════════════════════════════════════════════════════┘
                 │
                 ↓
        ┌────────────────────────────────┐
        │  matcher(outputs, targets)     │
        │                                │
        │  Computes cost matrix:         │
        │  • Class cost                  │
        │  • L1 box cost                 │
        │  • GIoU cost                   │
        │                                │
        │  Returns:                      │
        │  indices_final = [             │
        │      (src_idx_0, tgt_idx_0),   │  ← Batch 0: pred→target mapping
        │      (src_idx_1, tgt_idx_1),   │  ← Batch 1: pred→target mapping
        │      ...                       │
        │  ]                             │
        └────────┬─────────────────────────
                 │
                 ├═════════════════════════════════════════════════════┐
                 │  STEP 2: MATCH ALL AUXILIARY LAYERS                │
                 ├═════════════════════════════════════════════════════┘
                 │
                 ├→ matcher(aux_outputs[0], targets) → indices_aux[0]
                 │
                 ├→ matcher(aux_outputs[1], targets) → indices_aux[1]
                 │
                 ├→ matcher(aux_outputs[2], targets) → indices_aux[2]
                 │
                 ├→ matcher(aux_outputs[3], targets) → indices_aux[3]
                 │
                 └→ matcher(aux_outputs[4], targets) → indices_aux[4]
                       │
                       ↓
                 indices_aux_list = [
                     indices_aux[0],
                     indices_aux[1],
                     indices_aux[2],
                     indices_aux[3],
                     indices_aux[4],
                 ]
                 │
                 ├═════════════════════════════════════════════════════┐
                 │  STEP 3: COMPUTE MATCHING UNION (CONSENSUS)        │
                 ├═════════════════════════════════════════════════════┘
                 │
                 ↓
        ┌────────────────────────────────────┐
        │  matcher.compute_matching_union(   │
        │      indices_aux_list              │
        │  )                                 │
        │                                    │
        │  Algorithm:                        │
        │  For each (pred, tgt) pair:        │
        │    Count votes across 5 layers     │
        │    Keep high-consensus matches     │
        │                                    │
        │  Returns:                          │  ← Used for BOX losses
        │  indices_go = consensus indices    │     (more robust matching)
        └────────┬───────────────────────────┘
                 │
                 ├═════════════════════════════════════════════════════┐
                 │  STEP 4: COMPUTE FINAL LAYER LOSSES                │
                 ├═════════════════════════════════════════════════════┘
                 │
                 ↓
        ┌──────────────────────────────────────┐
        │  self.get_loss(                      │
        │      outputs,                        │
        │      targets,                        │
        │      indices_final,  ← Use final    │
        │      prefix=""                       │
        │  )                                   │
        │                                      │
        │  Returns:                            │
        │  {                                   │
        │      "loss_vfl": scalar,             │
        │      "loss_fgl": scalar              │
        │  }                                   │
        └────────┬─────────────────────────────┘
                 │
                 ├═════════════════════════════════════════════════════┐
                 │  STEP 5: COMPUTE AUX LAYER LOSSES (LOOP)          │
                 ├═════════════════════════════════════════════════════┘
                 │
                 ├→ for i=0: self.get_loss(aux_outputs[0], targets,
                 │                indices_aux[0], prefix="aux_0_")
                 │           → {"aux_0_loss_vfl": ..., "aux_0_loss_fgl": ...}
                 │
                 ├→ for i=1: self.get_loss(aux_outputs[1], targets,
                 │                indices_aux[1], prefix="aux_1_")
                 │           → {"aux_1_loss_vfl": ..., "aux_1_loss_fgl": ...}
                 │
                 ├→ for i=2: ...
                 │
                 ├→ for i=3: ...
                 │
                 └→ for i=4: ...
                       │
                 ├═════════════════════════════════════════════════════┐
                 │  STEP 6: AGGREGATE ALL LOSSES                      │
                 ├═════════════════════════════════════════════════════┘
                 │
                 ↓
        ┌────────────────────────────────────────────┐
        │  losses = {                                │
        │      "loss_vfl": final_vfl,                │
        │      "loss_fgl": final_fgl,                │
        │      "aux_0_loss_vfl": aux0_vfl,           │
        │      "aux_0_loss_fgl": aux0_fgl,           │
        │      "aux_1_loss_vfl": aux1_vfl,           │
        │      "aux_1_loss_fgl": aux1_fgl,           │
        │      ...                                   │
        │      "aux_4_loss_vfl": aux4_vfl,           │
        │      "aux_4_loss_fgl": aux4_fgl,           │
        │  }                                         │
        │                                            │
        │  Total: 2 + (5 * 2) = 12 loss values      │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
             return losses
```

---

## 4. Matching Union (GO Indices) Algorithm

```
┌──────────────────────────────────────────────────────────────────────┐
│              Matching Union: Consensus Across Layers                  │
│                    (GO = Global Oracle Indices)                       │
└──────────────────────────────────────────────────────────────────────┘

INPUT: List of matching indices from 5 auxiliary layers + final layer
       indices_list = [indices_0, indices_1, indices_2, indices_3, indices_4]

       Example for batch_size=1:
       
       indices_0 = [(torch.tensor([0,2,5]), torch.tensor([0,1,2]))]
       ↓
       Meaning: Image 0: Predictions {0,2,5} match targets {0,1,2}
       Query 0→Target 0, Query 2→Target 1, Query 5→Target 2

       indices_1 = [(torch.tensor([0,1,2,5]), torch.tensor([0,1,2,3]))]
       ↓  
       Query 0→Target 0, Query 1→Target 1, Query 2→Target 2, Query 5→Target 3

       ... (similarly for indices_2, indices_3, indices_4)


ALGORITHM:
──────────

Step 1: Count votes for each (query, target) pair
        ┌──────────────────────────────────────────────┐
        │  all_matches[batch_idx][query_id] = Counter  │
        │                                              │
        │  For each layer's indices:                   │
        │    For each (query, target) match:           │
        │      all_matches[batch_idx][query_id][tgt]++│
        │                                              │
        │  Result:                                     │
        │  all_matches[(0, 0)] = {0: 5}              │← Query 0 matches target 0 in all 5 layers
        │  all_matches[(0, 1)] = {1: 4, 2: 1}        │← Query 1: target 1 (4x), target 2 (1x)
        │  all_matches[(0, 2)] = {1: 3, 2: 2}        │← Query 2: target 1 (3x), target 2 (2x)
        │  all_matches[(0, 5)] = {2: 5}              │← Query 5: target 2 (5x)
        └──────────────────────────────────────────────┘

Step 2: Apply confidence filtering
        ┌──────────────────────────────────────────────────────┐
        │  threshold = num_layers * 0.5  (e.g., 5 * 0.5 = 2.5)│
        │                                                       │
        │  Keep only matches with vote count > threshold       │
        │                                                       │
        │  After filtering:                                    │
        │  all_matches[(0, 0)] = {0: 5}  ✓ (5 > 2.5)          │
        │  all_matches[(0, 1)] = {1: 4}  ✓ (4 > 2.5)          │
        │  all_matches[(0, 2)] = {1: 3}  ✓ (3 > 2.5)          │
        │  all_matches[(0, 5)] = {2: 5}  ✓ (5 > 2.5)          │
        └──────────────────────────────────────────────────────┘

Step 3: Resolve conflicts (query with multiple targets)
        ┌──────────────────────────────────────────────┐
        │  Query 1 has candidates: [1, 2]              │
        │    → Use highest vote count: target 1 (4 votes)      │
        │                                              │
        │  Query 2 has candidates: [1, 2]              │
        │    → Use highest vote count: target 1 (3 votes)      │
        │    → Conflict! Both query 1,2 want target 1  │
        │    → Keep query 1 (higher vote count)        │
        │    → Reassign query 2 to next best: target 2 │
        └──────────────────────────────────────────────┘

Step 4: Build final consensus indices
        ┌──────────────────────────────────────────────┐
        │  indices_go = [(                             │
        │      torch.tensor([0, 1, 2, 5]),            │
        │      torch.tensor([0, 1, 2, 2])             │
        │  )]                                          │
        │                                              │
        │  Final matches:                              │
        │  Query 0→Target 0 (5 votes, 100%)           │
        │  Query 1→Target 1 (4 votes, 80%)            │
        │  Query 2→Target 2 (3 votes, 60%)            │
        │  Query 5→Target 2 (5 votes, 100%)           │
        │                                              │
        │  Note: Deduplication removes duplicate       │
        │        target assignments                    │
        └──────────────────────────────────────────────┘


OUTPUT: consensus indices with high confidence
        Used for BOX REGRESSION LOSS (FGL)
        More robust than any single layer
```

---

## 5. Fine-Grained Localization (FGL) Loss Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│              Fine-Grained Localization Loss Computation               │
│          (Distribution-based bounding box regression)                │
└──────────────────────────────────────────────────────────────────────┘

                    INPUTS:
                    ───────
                    • pred_boxes: (B, N, 4)          [normalized 0-1]
                    • target_boxes: (M, 4)           [normalized 0-1]
                    • ref_points: (B, N, 2)          [box centers]
                    • pred_corners: (B, N, 4*33)     [logits]


    Step 1: Extract matched predictions & targets
    ────────────────────────────────────────────
            
            For each matched (query, target) pair:
            
            pred_box = pred_boxes[b, query_id]        • (4,)
            target_box = targets[b]["boxes"][tgt_id]  • (4,)
            ref_point = ref_points[b, query_id]       • (2,)
            
            Output:
            pred_boxes_matched    • (total_matched, 4)
            target_boxes_matched  • (total_matched, 4)
            ref_points_matched    • (total_matched, 2)


    Step 2: Convert target boxes to distance distributions
    ────────────────────────────────────────────────────
            
            For each target box:
            
            target_box = [x1, y1, x2, y2]
            ref_point  = [cx, cy]
            
            Compute edge distances:
            
            left   = cx - x1
            top    = cy - y1
            right  = x2 - cx
            bottom = y2 - cy
            
            distances = [left, top, right, bottom]
            
            ╔════════════════════════════════════════╗
            ║  For each distance d in distances:     ║
            ║                                        ║
            ║  1. Clamp to [0, reg_max]             ║
            ║     d = clamp(d, 0, 32)               ║
            ║                                        ║
            ║  2. Find integer bins: i = floor(d)   ║
            ║                        j = ceil(d)    ║
            ║                                        ║
            ║  3. Interpolate soft label:           ║
            ║     soft_label[i] = j - d             ║
            ║     soft_label[j] = d - i             ║
            ║                                        ║
            ║  4. Normalize (-> sums to 1)          ║
            ║                                        ║
            ║  Example: d = 5.7, reg_max=32         ║
            ║  soft_label = [0, 0, 0, 0, 0,        ║
            ║               0.3, 0.7, 0, ...]      ║
            ║               ↑────↑ (bins 5, 6)     ║
            ╚════════════════════════════════════════╝
            
            Output:
            soft_labels • (total_matched, 4, reg_max+1)
                         = (total_matched, 4, 33)


    Step 3: Reshape prediction logits to match soft labels
    ────────────────────────────────────────────────────
            
            pred_corners_matched: (total_matched, 4*33)
                                = (total_matched, 132)
            
            Reshape to: (total_matched*4, 33)
                      = (total_matched*4, 33)
            
            soft_labels: (total_matched, 4, 33)
            Reshape to: (total_matched*4, 33)
            
            Now both have compatible shapes!


    Step 4: Compute focal loss over distributions
    ───────────────────────────────────────────
            
            For each (distance, bin) pair:
            
            ╔════════════════════════════════════════════════╗
            ║  1. Convert logits to probabilities:          ║
            ║     prob = softmax(pred_dist, dim=-1)        ║
            ║                                               ║
            ║  2. Compute focal weight:                     ║
            ║     weight = (1 - prob) ^ gamma              ║
            ║     (focus on hard cases, low probability)   ║
            ║                                               ║
            ║  3. Compute cross-entropy:                    ║
            ║     ce = -log(prob)                          ║
            ║                                               ║
            ║  4. Apply focal weighting:                    ║
            ║     focal_loss = weight * ce * soft_label    ║
            ║                                               ║
            ║  5. Sum over all bins:                        ║
            ║     loss_per_example = sum(focal_loss)       ║
            ║                                               ║
            ║  Example: If soft_label peaks at bin 6:      ║
            ║  • High prob at bin 6 → weight ≈ 0            ║
            ║  • Low prob at bin 3  → weight ≈ 1            ║
            ║  → Focus training on fixing bin 3             ║
            ╚════════════════════════════════════════════════╝
            
            Output:
            loss_per_example • (total_matched*4,)


    Step 5: Apply quality weighting (optional)
    ───────────────────────────────────────
            
            iou_scores = compute_iou(pred_boxes, target_boxes)
            
            Easy boxes (high IoU):   weight ≈ 0.5  (low loss)
            Hard boxes (low IoU):    weight ≈ 1.0  (high loss)
            
            weighted_loss = loss_per_example * iou_scores
            
            Output:
            weighted_loss • (total_matched*4,)


    Step 6: Aggregate to final FGL loss
    ────────────────────────────────
            
            FGL_loss = weighted_loss.mean()
            
            Output:
            FGL_loss • scalar tensor (differentiable)


    FINAL OUTPUT:
    ─────────────
    FGL_loss: scalar that reflects "How close are predicted
              distance distributions to target boxes?"
              
              • Low:  Predictions match targets well
              • High: Predictions are far from targets
```

---

## 6. Loss Computation per Layer (get_loss)

```
┌──────────────────────────────────────────────────────────────────────┐
│           Computing VFL + FGL Loss for a Single Layer                │
│                   (Called 6 times: final + 5 aux)                    │
└──────────────────────────────────────────────────────────────────────┘

INPUT:
───────
• layer_outputs: {pred_logits, pred_boxes, pred_corners}
• targets: List[Dict] with labels, boxes
• indices: Matched (query, target) pairs for this layer
• prefix: Loss name prefix ("" for final, "aux_0_" for layer 0, etc.)


VFL LOSS (Classification)
──────────────────────────

    ┌─────────────────────────────────────────────┐
    │  Extract matched predictions & targets      │
    └─────────────────────────────────────────────┘
         │
         ↓
    ┌────────────────────────────────────────────────┐
    │  For each matched pair (query, target):        │
    │                                                │
    │  pred_logits_matched ← outputs[b, query_id]   │
    │                        (shape: C for 80 class)│
    │                                                │
    │  target_label ← targets[b]["labels"][tgt_id]  │
    │                (shape: scalar 0-79)           │
    └────────────┬─────────────────────────────────┘
                 │
                 ↓
    ┌────────────────────────────────────────────────┐
    │  Stack across all matched pairs:               │
    │                                                │
    │  pred_logits_all: (total_matched, 80)         │
    │  target_labels_all: (total_matched,)          │
    └────────────┬─────────────────────────────────┘
                 │
                 ↓
    ┌────────────────────────────────────────────────┐
    │  Create one-hot target scores:                 │
    │                                                │
    │  target_scores[i, target_labels_all[i]] = 1.0 │
    │  (rest are 0)                                  │
    │                                                │
    │  target_scores: (total_matched, 80)           │
    └────────────┬─────────────────────────────────┘
                 │
                 ↓
    ┌────────────────────────────────────────────────┐
    │  Call varifocal_loss():                        │
    │                                                │
    │  loss_vfl = varifocal_loss(                    │
    │      pred_logits_all,                          │
    │      target_scores                             │
    │  )                                             │
    │                                                │
    │  Result: scalar (differentiable)              │
    └────────────┬─────────────────────────────────┘
                 │
                 ↓
    ┌────────────────────────────────────────────────┐
    │  Apply weight from weight_dict:                │
    │                                                │
    │  weighted_loss = loss_vfl *                    │
    │                  weight_dict["loss_vfl"]      │
    │                                                │
    │  (Typically weight_dict["loss_vfl"] = 1.0)    │
    └────────────┬─────────────────────────────────┘
                 │
                 ↓
    layer_losses[f"{prefix}loss_vfl"] = weighted_loss


FGL LOSS (Localization)
───────────────────────

    ┌─────────────────────────────────────────────┐
    │  Extract matched predictions & targets      │
    └─────────────────────────────────────────────┘
         │
         ├─→ pred_boxes_matched ← outputs[b, query_id, :]
         │                        (shape: B, N, 4)
         │
         ├─→ pred_corners_matched ← outputs[b, query_id, :]
         │                          (shape: B, N, 4*33)
         │
         ├─→ target_boxes_matched ← targets[b]["boxes"][tgt_id, :]
         │                          (shape: M, 4)
         │
         └─→ ref_points_matched ← box centers from pred_boxes
                                  (shape: B, N, 2)
         │
         ↓
    ┌───────────────────────────────────────────────────────┐
    │  Call bbox2distance():                                │
    │                                                       │
    │  soft_labels, weight_per_side =                       │
    │      bbox2distance(                                   │
    │          ref_points_matched,    # Centers            │
    │          target_boxes_matched,  # Ground truth        │
    │          reg_max=32             # Num bins            │
    │      )                                                │
    │                                                       │
    │  soft_labels: (total_matched, 4, 33)                │
    │  weight_per_side: (total_matched, 4)                │
    └────────────┬──────────────────────────────────────────┘
                 │
                 ↓
    ┌──────────────────────────────────────────────┐
    │  Reshape for focal loss computation:         │
    │                                              │
    │  pred_corners_reshaped:                      │
    │    (total_matched, 4*33)                     │
    │    → (total_matched*4, 33)                   │
    │                                              │
    │  soft_labels_flat:                           │
    │    (total_matched, 4, 33)                    │
    │    → (total_matched*4, 33)                   │
    └────────────┬──────────────────────────────────┘
                 │
                 ↓
    ┌──────────────────────────────────────────────┐
    │  Call unimodal_distribution_focal_loss():    │
    │                                              │
    │  loss_fgl = unimodal_distribution_focal_loss(│
    │      pred_corners_flat,                      │
    │      soft_labels_flat,                       │
    │      weight=combined_weights,                │
    │      reduction="mean"                        │
    │  )                                           │
    │                                              │
    │  Result: scalar (differentiable)            │
    └────────────┬──────────────────────────────────┘
                 │
                 ↓
    ┌──────────────────────────────────────────────┐
    │  Apply weight from weight_dict:              │
    │                                              │
    │  weighted_loss = loss_fgl *                  │
    │                  weight_dict["loss_fgl"]    │
    │                                              │
    │  (Typically weight_dict["loss_fgl"] = 5.0)  │
    └────────────┬──────────────────────────────────┘
                 │
                 ↓
    layer_losses[f"{prefix}loss_fgl"] = weighted_loss


OUTPUT:
───────
layer_losses = {
    f"{prefix}loss_vfl": scalar,   # Only if "vfl" in losses list
    f"{prefix}loss_fgl": scalar,   # Only if "fgl" in losses list
}

Example for final layer (prefix=""):
{
    "loss_vfl": 1.234,
    "loss_fgl": 0.567
}

Example for auxiliary layer 0 (prefix="aux_0_"):
{
    "aux_0_loss_vfl": 1.345,
    "aux_0_loss_fgl": 0.789
}
```

---

## 7. Model Output Dictionary Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Complete Model Output Format                       │
│                          (from model.forward())                       │
└──────────────────────────────────────────────────────────────────────┘

outputs = {
    # ─── FINAL LAYER PREDICTIONS (Decoder Layer 5) ─────────────────
    "pred_logits": Tensor,         # Shape: (B, N, num_classes)
                                   #        (2, 50, 80)
                                   # Content: Class logits
                                   # Use: VFL loss
                                   
    "pred_boxes": Tensor,          # Shape: (B, N, 4)
                                   #        (2, 50, 4)
                                   # Range: [0, 1] (sigmoid normalized)
                                   # Format: [x1, y1, x2, y2]
                                   # Use: Box regression, matching
                                   
    "pred_corners": Tensor,        # Shape: (B, N, 4*(reg_max+1))
                                   #        (2, 50, 132)
                                   # Content: Distance distribution logits
                                   # 4: number of edges (L, T, R, B)
                                   # 33: bins (0-32 inclusive)
                                   # Use: FGL loss
                                   
    "ref_points": Tensor,          # Shape: (B, N, 2)
                                   #        (2, 50, 2)
                                   # Content: Box center coordinates
                                   # Use: Reference for FGL distance conversion
                                   
    # ─── AUXILIARY LAYER PREDICTIONS (Decoder Layers 0-4) ──────────
    "aux_outputs": List[Dict],     # Length: 5 (one per auxiliary layer)
                                   # Each dict contains:
                                   
        [{
            "pred_logits": Tensor,   # (B, N, C)
            "pred_boxes": Tensor,    # (B, N, 4)
            "pred_corners": Tensor,  # (B, N, 4*33)
        },
        ...  # 5 times total
        ]
    
    # ─── ENCODER AUXILIARY (Optional) ──────────────────────────────
    "enc_aux_outputs": List[Dict], # Encoder intermediate outputs
                                   # (rarely used in loss computation)
    
    # ─── INTERMEDIATE OUTPUTS ────────────────────────────────────
    "pre_outputs": Dict,           # Pre-refined box predictions
                                   # Contains: pred_logits, pred_boxes, pred_corners
    
    # ─── SCALE & CALIBRATION FACTORS ─────────────────────────────
    "reg_scale": Tensor,           # Shape: (B, N) or (1,)
                                   # Scaling factors for FGL
    
    "up": Tensor,                  # Shape: (B, N) or (1,)
                                   # Upscale factors for FGL
}

USAGE IN CRITERION:
───────────────────

Forward pass to criterion:
    losses = criterion(outputs, targets)
    
    Criterion extracts:
    • Final layer: pred_logits, pred_boxes, pred_corners
    • Aux layers: outputs["aux_outputs"][i] for each i
    
    Uses each for separate matching + loss computation

SHAPES REFERENCE:
─────────────────

(B, N, C)     = (batch_size, num_queries, num_classes)
              = (2, 50, 80)

(B, N, 4)     = (batch_size, num_queries, 4)
              = (2, 50, 4)

(B, N, 4*33)  = (batch_size, num_queries, 4*reg_max+1)
              = (2, 50, 132)
              = 4 edges × 33 distance bins per edge

Matching indices shape:
                [(src_idx, tgt_idx),  ← Per batch item
                 (src_idx, tgt_idx),
                 ...]
                
    src_idx: (num_matched,) - which queries matched
    tgt_idx: (num_matched,) - which targets matched
```

---

## 8. Training Loop Flowchart

```
┌──────────────────────────────────────────────────────────────────────┐
│                  D-FINE Training Loop Flowchart                       │
└──────────────────────────────────────────────────────────────────────┘

    Initialize:
    • model = DFINEMini(...)          ← Multi-layer model
    • matcher = HungarianMatcher()
    • criterion = DFINECriterion(matcher, ...)  ← Multi-layer criterion
    • optimizer = AdamW(model.parameters())
    
    
FOR each epoch:
──────────────

    FOR each batch (images, targets) in dataloader:
    ───────────────────────────────────────────────
    
        ┌─────────────────────────────────────────┐
        │ FORWARD PASS                            │
        └─────────────────────────────────────────┘
        
        outputs = model(images)
        │
        ├─→ outputs["pred_logits"]: (B, N, 80)
        ├─→ outputs["pred_boxes"]: (B, N, 4)
        ├─→ outputs["pred_corners"]: (B, N, 132)
        ├─→ outputs["ref_points"]: (B, N, 2)
        └─→ outputs["aux_outputs"]: List[5 dicts]
        
        ┌─────────────────────────────────────────┐
        │ LOSS COMPUTATION (Multi-layer)          │
        ├─────────────────────────────────────────┤
        │ DFINECriterion internally:              │
        │  1. Matches final layer (6 times)       │
        │  2. Matches aux layers (5 times)        │
        │  3. Computes matching union consensus   │
        │  4. Aggregates 12 loss values           │
        └─────────────────────────────────────────┘
        
        losses = criterion(outputs, targets)
        │
        ├─→ losses["loss_vfl"]: scalar
        ├─→ losses["loss_fgl"]: scalar
        ├─→ losses["aux_0_loss_vfl"]: scalar
        ├─→ losses["aux_0_loss_fgl"]: scalar
        ├─→ ...
        └─→ losses["aux_4_loss_fgl"]: scalar
        
        ┌─────────────────────────────────────────┐
        │ AGGREGATE LOSSES                        │
        └─────────────────────────────────────────┘
        
        total_loss = sum(losses.values())
        │
        └─→ total_loss: scalar (6 final + 5×2 aux values combined)
        
        ┌─────────────────────────────────────────┐
        │ BACKWARD PASS                           │
        └─────────────────────────────────────────┘
        
        optimizer.zero_grad()
        total_loss.backward()
        │
        ├─→ Gradients flow through:
        │   • Criterion (matching, loss computation)
        │   • Model (decoder, encoder, backbone)
        │   • All 6 decoder layers receive supervision
        │   • All parameters get ∂loss/∂param
        
        ┌─────────────────────────────────────────┐
        │ OPTIMIZATION STEP                       │
        └─────────────────────────────────────────┘
        
        optimizer.step()
        │
        ├─→ Updates all model parameters:
        │   param ← param - lr * param.grad
        │   
        │   Note: All layers (0-5) improved, not just final!
        │   This is multi-layer supervision in action.
        
        ┌─────────────────────────────────────────┐
        │ LOGGING                                 │
        └─────────────────────────────────────────┘
        
        print(f"Epoch {epoch}: total_loss={total_loss:.4f}")
        print(f"  loss_vfl={losses['loss_vfl']:.4f}")
        print(f"  loss_fgl={losses['loss_fgl']:.4f}")
        print(f"  aux_avg={sum([l for k,l in losses.items() if 'aux' in k])/10:.4f}")


END epoch
─────────

scheduler.step()
│
└─→ Learning rate decay (e.g., StepLR every 50 epochs)
```

---

## 9. Tensor Flow Diagram: Single Training Iteration

```
┌─────────────────────────────────────────────────────────────────┐
│        Tensor Dimensions Through Pipeline (1 Iteration)        │
└─────────────────────────────────────────────────────────────────┘

INPUT
─────
images: (B=2, C=3, H=256, W=256)
targets: [
    {labels: (M1,), boxes: (M1, 4)},   ← Image 0: 3 objects
    {labels: (M2,), boxes: (M2, 4)}    ← Image 1: 2 objects
]

┌────────────────┐
│   BACKBONE     │
├────────────────┤
│ HGNetV2        │
│ Extract multi  │
│ feature levels │
└────────┬───────┘

Output: features at multiple scales
    feat_c3: (B, 128, 64, 64)   ← Scale 1/4
    feat_c4: (B, 256, 32, 32)   ← Scale 1/8
    feat_c5: (B, 512, 16, 16)   ← Scale 1/16

┌────────────────┐
│   MULTI-LEVEL  │
│   NECK (FPN)   │
├────────────────┤
│ Combine scales │
│ Top-down path  │
│ Lateral conns  │
└────────┬───────┘

Output: 4 feature levels
    p3: (B, 128, 64, 64)
    p4: (B, 128, 32, 32)
    p5: (B, 128, 16, 16)
    p6: (B, 128, 8, 8)
All feature levels now aligned to 128 dims!

┌────────────────┐
│   FLATTEN &    │
│   CONCATENATE  │
├────────────────┤
│ Flatten all    │
│ spatial dims   │
└────────┬───────┘

feats_flat: (B, C=256*128, HW=4096)
    = (2, 32768, 4096)
    [Batch, Channels, Spatial Tokens]

┌────────────────┐
│   ENCODER      │
│   (6 Layers)   │
├────────────────┤
│ Self-attention │
│ over tokens    │
└────────┬───────┘

enc_output: (B, L=4096, D=256)
    Each spatial position attends to all others
    All 6 encoder layers process sequentially

┌────────────────┐
│   OBJECT       │
│   QUERIES      │
│   + PE         │
├────────────────┤
│ Learnable      │
│ query vectors  │
│ + sin/cos PE   │
└────────┬───────┘

queries: (B, N=50, D=256)
    50 query embeddings (learnable)

┌────────────────────────────────────────────┐
│   DECODER (6 Stacked Layers)               │
│                                            │
│   For layer_idx in range(6):               │
│   ├ queries input shape: (B, 50, 256)     │
│   │                                        │
│   ├ Cross-attn with encoder                │
│   │  queries·encoder → attention weights   │
│   │  Output: (B, 50, 256) attended         │
│   │                                        │
│   ├ Self-attn among queries                │
│   │  queries·queries                       │
│   │  Output: (B, 50, 256) refined          │
│   │                                        │
│   ├ Feed-forward                           │
│   │  Linear→ReLU→Linear                    │
│   │  Output: (B, 50, 256) final layer out  │
│   │                                        │
│   └─→ Prediction head (layer_idx < 6):    │
│       ├ pred_logits_i: (B, 50, 80)        │
│       ├ pred_boxes_i: (B, 50, 4)          │
│       └ pred_corners_i: (B, 50, 132)      │
│                                            │
│       Total auxiliary outputs: 5 sets      │
└────────┬───────────────────────────────────┘
         │
┌────────▼─────────────────────────────────┐
│   FINAL LAYER OUTPUT (Layer 5)           │
├──────────────────────────────────────────┤
│ pred_logits_final: (B=2, N=50, C=80)    │
│ pred_boxes_final:  (B=2, N=50, 4)       │
│ pred_corners_fin:  (B=2, N=50, 132)     │
│ ref_points_final:  (B=2, N=50, 2)       │
│                                          │
│ aux_outputs: List of 5 dicts            │
│   [0]: {logits: (B,50,80), boxes: ...}  │
│   [1]: {logits: (B,50,80), boxes: ...}  │
│   [2]: {logits: (B,50,80), boxes: ...}  │
│   [3]: {logits: (B,50,80), boxes: ...}  │
│   [4]: {logits: (B,50,80), boxes: ...}  │
└────────┬─────────────────────────────────┘
         │
┌────────▼──────────────────────────┐
│   CRITERION (Multi-layer)         │
│                                   │
│   For final + 5 aux layers:       │
│                                   │
│   matcher(layer_output, targets)  │
│   → indices: [(src, tgt), ...]    │
│                                   │
│   Extract matched predictions:    │
│   • logits_matched: (total, C)    │
│   • boxes_matched: (total, 4)     │
│   • corners_matched: (total, 132) │
│                                   │
│   Extract matched targets:        │
│   • labels_matched: (total,)      │
│   • boxes_matched_gt: (total, 4)  │
│                                   │
│   Compute losses:                 │
│   • VFL: logits vs labels         │
│   • FGL: corners vs box_dists     │
└────────┬──────────────────────────┘
         │
┌────────▼──────────────────────────────────┐
│   LOSS AGGREGATION                        │
│                                           │
│   loss_dict = {                           │
│       "loss_vfl": scalar,                 │
│       "loss_fgl": scalar,                 │
│       "aux_0_loss_vfl": scalar,           │
│       "aux_0_loss_fgl": scalar,           │
│       "aux_1_loss_vfl": scalar,           │
│       "aux_1_loss_fgl": scalar,           │
│       ...                                 │
│       "aux_4_loss_vfl": scalar,           │
│       "aux_4_loss_fgl": scalar            │
│   }  [12 values total]                    │
│                                           │
│   total_loss = sum(loss_dict.values())    │
│              = scalar                     │
└────────┬──────────────────────────────────┘
         │
┌────────▼──────────────────────┐
│   BACKPROPAGATION             │
│                               │
│   total_loss.backward()       │
│                               │
│   Computes gradients:         │
│   • ∂total/∂(layer5_logits)  │
│   • ∂total/∂(layer4_logits)  │
│   • ...                       │
│   • ∂total/∂(layer0_logits)  │
│   • ∂total/∂(encoder_params) │
│   • ∂total/∂(backbone_params)│
│                               │
│   All layers get supervision! │
└────────┬──────────────────────┘
         │
┌────────▼──────────────────────┐
│   OPTIMIZER STEP              │
│                               │
│   optimizer.step()            │
│                               │
│   Updates all parameters      │
│   with learning rate × grad   │
│                               │
│   Model improved from all     │
│   6 layers simultaneously!    │
└───────────────────────────────┘
```

---

## Summary Table: Key Components & Shapes

| Component | Input Shape | Output Shape | Purpose |
|---|---|---|---|
| **Backbone** | (B,3,H,W) | Multi-scale feats | Feature extraction |
| **Neck** | Backbone feats | (B,128,64×64), (B,128,32×32), (B,128,16×16), (B,128,8×8) | Align feature scales |
| **Encoder** | (B, 4096, 256) | (B, 4096, 256) | Semantic encoding |
| **Decoder×6** | (B, 50, 256) + encoder | (B, 50, 256) | Object queries |
| **Pred Heads×6** | (B, 50, 256) | Logits: (B,50,80), Boxes: (B,50,4), Corners: (B,50,132) | Final predictions |
| **Matcher×6** | Predictions + Targets | Indices: List[(src, tgt)] | Hungarian matching |
| **VFL Loss** | Logits: (total,80), Labels: (total,) | scalar | Classification |
| **FGL Loss** | Corners: (total,132), Boxes: (total,4) | scalar | Localization |

These diagrams provide complete visual reference for understanding and debugging the D-FINE architecture!
