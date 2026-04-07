# Problem 11 Questions

## Multiple Choice

1. What is the purpose of the 1×1 projection convolution between HGNetV2 backbone and the encoder?
   A. To double the spatial resolution
   B. To match the backbone's output channel count (96) to the transformer d_model (256)
   C. To apply BatchNorm after the backbone
   D. To reduce the spatial size before the transformer

2. How does the decoder receive spatial positional information for the encoder memory?
   A. The encoder output already encodes positions — no additional PE needed
   B. The same Sine 2D PE used in the encoder is passed as `memory_pos` to decoder cross-attention
   C. The decoder uses learnable PE independently
   D. Positions are embedded in the object queries

3. Why must `pred_boxes` always be passed through `sigmoid()` before returning?
   A. Sigmoid prevents the box regression loss from being too large
   B. Sigmoid maps raw logits to (0,1), ensuring boxes stay within the normalized image coordinate range
   C. The Hungarian Matcher requires sigmoid-normalized inputs
   D. It's required by the GIoU loss computation

4. In the decoder, `query_embed` and `query_pos` serve different roles:
   A. `query_embed` is the initial content of queries; `query_pos` is the positional anchor for cross-attention routing
   B. They are interchangeable — both encode position
   C. `query_pos` encodes class labels; `query_embed` encodes spatial position
   D. Only one of them is needed; the other is redundant

5. If the input image is `(B, 3, 256, 256)`, what is the sequence length T fed into the Transformer Encoder?
   A. `256 × 256 = 65,536`
   B. `128 × 128 = 16,384`
   C. `32 × 32 = 1,024`
   D. `64 × 64 = 4,096`

## Answer Key
1.B 2.B 3.B 4.A 5.C
