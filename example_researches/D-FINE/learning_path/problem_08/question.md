# Problem 08 Questions

## Multiple Choice

1. In a Pre-LN Transformer encoder layer, LayerNorm is applied:
   A. After the residual addition (output of the sub-layer)
   B. Before the attention/FFN computation (at the beginning of the residual branch)
   C. Only on the value vectors
   D. After every Linear layer

2. Why is positional encoding added to q and k but NOT to v in D-FINE's encoder?
   A. Because v has a different shape
   B. To inject spatial routing information into attention without corrupting the value content being aggregated
   C. To reduce memory use
   D. It is a convention with no mathematical justification

3. The standard FFN expansion ratio in transformers is typically:
   A. d_ffn = d_model (no expansion)
   B. d_ffn = 2 * d_model
   C. d_ffn = 4 * d_model
   D. d_ffn = 8 * d_model

4. What activation is commonly used in modern transformer FFN (including D-FINE)?
   A. ReLU
   B. Sigmoid
   C. GELU
   D. Tanh

5. D-FINE's Hybrid Encoder applies encoder layers to feature maps from which backbone stages?
   A. Only the final, deepest stage
   B. All 4 backbone stages equally
   C. Multiple scales (small, medium, large) simultaneously with uneven sampling
   D. None — the encoder only applies conv layers

## Answer Key
1.B 2.B 3.C 4.C 5.C
