# Problem 07 Theory - Full DETR Assembly

## Core Definitions
- DETR composes backbone, positional encoding, transformer, and prediction heads into one end-to-end detector.
- Predictions are fixed-size set outputs with one no-object class.

## Variables and Shape Dictionary
- $B$: batch size.
- $H,W$: input image size.
- $N_q$: number of object queries.
- $C$: hidden dimension.
- $K$: foreground class count.
- Input image: $(B,3,H,W)$.
- Decoder output: $(B,N_q,C)$.
- Class logits: $(B,N_q,K+1)$.
- Boxes: $(B,N_q,4)$.

## Main Equations (LaTeX)
$$
F = \text{Backbone}(X), \quad Z = W_{proj}*F + P
$$
$$
H = \text{Transformer}(Z, Q_o)
$$
$$
Y_{cls}=H W_c + b_c, \quad Y_{box}=\sigma(\text{MLP}(H))
$$

## Step-by-Step Derivation or Computation Flow
1. Extract CNN features from image.
2. Project channels to transformer width and add positional context.
3. Run encoder-decoder transformer with learned object queries.
4. Decode each query into class logits and box coordinates.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input image: $(B,3,H,W)$.
- Backbone map: $(B,C_b,H',W')$.
- Flattened tokens: $(H'W',B,C)$.
- Decoder states: $(B,N_q,C)$.
- Outputs: class $(B,N_q,K+1)$, box $(B,N_q,4)$.

Worked mini-example:
- Let $B=2$, $H=W=640$, $H'=W'=20$, $N_q=100$, $C=256$, $K=80$.
- Logits shape: $(2,100,81)$ and boxes shape: $(2,100,4)$.

## Practical Interpretation
- This assembly removes proposal/NMS stages and trains all components jointly.
- Query-based parallel prediction is the core design that simplifies detection pipelines.