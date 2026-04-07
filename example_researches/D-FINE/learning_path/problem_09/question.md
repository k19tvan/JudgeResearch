# Problem 09 Questions

## Multiple Choice

1. What is the key structural difference between D-FINE's decoder layer and a standard DETR decoder layer?
   A. D-FINE uses ReLU instead of GELU
   B. D-FINE replaces the cross-attention residual connection with a Target Gating Layer (TGL)
   C. D-FINE uses 16 attention heads by default
   D. D-FINE removes the self-attention sub-layer

2. In the Target Gating Layer, why does the gate input use `cat([tgt, cross_out], dim=-1)`?
   A. To double the output dimension permanently
   B. To allow each gate to consider BOTH the current query state and the new cross-attended content for its decision
   C. Concatenation is required by sigmoid
   D. To combine spatial and semantic features

3. In D-FINE's decoder cross-attention, what plays the role of Query, Key, and Value?
   A. Q=image features, K=V=object queries
   B. Q=object queries+qpos, K=memory+mpos, V=memory (no pos for V)
   C. Q=K=V=object queries
   D. Q=memory, K=V=object queries+qpos

4. The Target Gating Layer from the ablation study (Table 3 of the paper) recovers AP from:
   A. 53.0% → 54.0%
   B. 52.4% → 52.8%
   C. 50.0% → 53.0%
   D. 51.0% → 52.4%

5. Why does the Transformer Decoder have both self-attention AND cross-attention?
   A. Self-attention only refines features; cross-attention generates queries
   B. Self-attention suppresses duplicate predictions (queries inhibit each other); cross-attention fetches object-specific image evidence
   C. Both are needed to handle variable image sizes
   D. Self-attention handles classification; cross-attention handles box regression

## Answer Key
1.B 2.B 3.B 4.B 5.B
