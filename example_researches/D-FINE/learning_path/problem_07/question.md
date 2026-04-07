# Problem 07 Questions

## Multiple Choice

1. Why is the attention score divided by `sqrt(d_k)` before softmax?
   A. To ensure values stay in [0,1]
   B. To normalize the number of heads
   C. To prevent the dot product variance from growing with d_k, which would push softmax into saturation
   D. To make attention symmetric

2. In Multi-Head Attention, what is the purpose of using multiple heads instead of one?
   A. To reduce computation by parallelism
   B. To allow the model to attend to different representation subspaces simultaneously
   C. To apply different dropout masks
   D. To process different batch elements

3. After computing the context `ctx = attn @ V` of shape `(B, H, T_q, d_k)`, what is the correct reshape to `(B, T_q, d_model)`?
   A. `.reshape(B, T_q, d_model)` directly
   B. `.transpose(1,2).contiguous().reshape(B, T_q, d_model)`
   C. `.flatten(1)`
   D. `.permute(0,2,1,3).sum(-1)`

4. What does setting `key_padding_mask[b, t] = True` do to position `t` in image `b`?
   A. Amplifies its attention weight
   B. The corresponding attention score is set to `-inf` before softmax, resulting in ~0 weight
   C. The value vector is zeroed out
   D. The key projection is skipped

5. In D-FINE's decoder, cross-attention uses:
   A. `q = k = v = image features`
   B. `q = object queries`, `k = v = image features`
   C. `q = image features`, `k = v = object queries`
   D. `q = k = object queries`, `v = image features`

## Answer Key
1.C 2.B 3.B 4.B 5.B
