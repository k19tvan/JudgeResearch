# DETR Theory Reference

This document contains the theory for DETR in a dedicated place, with explicit equations, tensor shapes, and variable semantics.

## 1. Problem Formulation: Detection as Set Prediction

DETR formulates object detection as predicting a fixed-size set of detections from an image.

Given an image $x$, the model predicts a set:

$$
\hat{Y} = \{\hat{y}_i\}_{i=1}^{N_q}, \quad \hat{y}_i = (\hat{p}_i, \hat{b}_i)
$$

where:
- $N_q$: number of object queries (default: $100$).
- $\hat{p}_i \in [0,1]^{K+1}$: class probability vector for query $i$.
- $K$: number of foreground classes.
- $+1$: the no-object class.
- $\hat{b}_i = (\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h}) \in [0,1]^4$: normalized bounding box.

Shape contract:
- Classification logits before softmax: $(B, N_q, K+1)$.
- Boxes: $(B, N_q, 4)$.

## 2. Backbone Features and Positional Encoding

For image batch $x \in \mathbb{R}^{B \times 3 \times H \times W}$, backbone output is:

$$
f = \text{Backbone}(x) \in \mathbb{R}^{B \times C \times H' \times W'}
$$

where:
- $H' = H / s$, $W' = W / s$ with stride $s=32$ for standard ResNet C5 output.
- $C=2048$ for ResNet-50/101 final stage.

A $1\times1$ projection maps channels to transformer width $d$:

$$
z = W_{proj} * f \in \mathbb{R}^{B \times d \times H' \times W'}, \quad d=256
$$

Positional encoding $p \in \mathbb{R}^{B \times d \times H' \times W'}$ is added in attention computations.

Flatten to sequence length $L = H'W'$:
- $z_{seq} \in \mathbb{R}^{L \times B \times d}$
- $p_{seq} \in \mathbb{R}^{L \times B \times d}$

## 3. Transformer Encoder-Decoder Theory

### 3.1 Encoder Self-Attention

For each head:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V
$$

where:
- $Q,K,V \in \mathbb{R}^{L \times d_h}$ per sample and head.
- $d_h = d / h$ with $h$ heads.
- $M$: mask bias (large negative value for padded positions).

Encoder output memory:

$$
\text{memory} \in \mathbb{R}^{L \times B \times d}
$$

### 3.2 Decoder with Object Queries

Learned object queries:

$$
Q_o \in \mathbb{R}^{N_q \times d}
$$

Broadcast for batch:

$$
Q_o^{(B)} \in \mathbb{R}^{N_q \times B \times d}
$$

Decoder performs:
1. Query self-attention over $N_q$ slots.
2. Cross-attention from queries to encoder memory.
3. FFN refinement.

Final decoder states:

$$
H_{dec}^{(\ell)} \in \mathbb{R}^{B \times N_q \times d}, \quad \ell=1,\dots,L_d
$$

where $L_d$ is number of decoder layers (default: $6$).

## 4. Prediction Heads

### 4.1 Classification Head

Linear projection to class logits:

$$
\text{logits} = H_{dec} W_c + b_c \in \mathbb{R}^{B \times N_q \times (K+1)}
$$

Class probabilities:

$$
\hat{p}_{i} = \text{softmax}(\text{logits}_{i})
$$

### 4.2 Box Regression Head

An MLP predicts 4 normalized coordinates:

$$
\hat{b}_i = \sigma(\text{MLP}(h_i)) = (\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h})
$$

where:
- $\sigma$: sigmoid, constraining values to $[0,1]$.
- $h_i \in \mathbb{R}^{d}$ is query feature for slot $i$.

## 5. Bipartite Matching (Hungarian)

Let ground truth set for sample $n$ be:

$$
Y^{(n)} = \{(c_j, b_j)\}_{j=1}^{N_{gt}}
$$

DETR solves one-to-one assignment by minimizing:

$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_{N_q}} \sum_{j=1}^{N_{gt}} \mathcal{C}_{\text{match}}\big(y_j, \hat{y}_{\sigma(j)}\big)
$$

with matching cost:

$$
\mathcal{C}_{\text{match}} = \lambda_{cls}\,\mathcal{C}_{cls} + \lambda_{L1}\,\mathcal{C}_{L1} + \lambda_{giou}\,\mathcal{C}_{giou}
$$

Typical terms:
- $\mathcal{C}_{cls} = -\hat{p}_{\sigma(j)}(c_j)$.
- $\mathcal{C}_{L1} = \|b_j - \hat{b}_{\sigma(j)}\|_1$.
- $\mathcal{C}_{giou} = 1 - \text{GIoU}(b_j, \hat{b}_{\sigma(j)})$.

Variable meaning:
- $\sigma$: permutation mapping GT objects to unique predictions.
- $N_q$: number of predictions (queries), usually greater than $N_{gt}$.

## 6. Training Loss

After optimal assignment, loss is computed on matched pairs:

$$
\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{L1}\mathcal{L}_{bbox} + \lambda_{giou}\mathcal{L}_{giou}
$$

### 6.1 Classification Loss

$$
\mathcal{L}_{cls} = -\sum_{i=1}^{N_q} w_{c_i} \log p_i(c_i)
$$

where:
- $c_i$ is target class for query $i$ (foreground or no-object).
- $w_{c_i}$ includes reduced weight for no-object (eos coefficient).

### 6.2 Box Regression Loss

$$
\mathcal{L}_{bbox} = \frac{1}{N_{gt}}\sum_{(i,j)\in \hat{\sigma}} \|\hat{b}_i - b_j\|_1
$$

### 6.3 GIoU Loss

$$
\mathcal{L}_{giou} = \frac{1}{N_{gt}}\sum_{(i,j)\in \hat{\sigma}} \big(1 - \text{GIoU}(\hat{b}_i, b_j)\big)
$$

### 6.4 Auxiliary Decoder Losses

If decoder returns intermediate layers, apply the same losses at each layer:

$$
\mathcal{L}_{total} = \sum_{\ell=1}^{L_d} \alpha_\ell\,\mathcal{L}^{(\ell)}
$$

where:
- $\mathcal{L}^{(\ell)}$: DETR loss from decoder layer $\ell$.
- $\alpha_\ell$: auxiliary loss weight.

## 7. Inference Semantics

Given outputs:
- logits: $(B, N_q, K+1)$
- boxes: $(B, N_q, 4)$

Inference applies:
1. Softmax over class dimension.
2. Remove no-object-dominant predictions or threshold by confidence.
3. Convert box from $(c_x, c_y, w, h)$ to $(x_{min}, y_{min}, x_{max}, y_{max})$ in pixels.

Coordinate conversion for image size $(H_{img}, W_{img})$:

$$
\begin{aligned}
x_{min} &= (c_x - w/2)W_{img} \\
y_{min} &= (c_y - h/2)H_{img} \\
x_{max} &= (c_x + w/2)W_{img} \\
y_{max} &= (c_y + h/2)H_{img}
\end{aligned}
$$

## 8. Symbol and Shape Dictionary

- $B$: batch size.
- $H, W$: input image height and width.
- $H', W'$: backbone feature spatial size.
- $C$: backbone channel dimension.
- $d$: transformer hidden size.
- $L=H'W'$: flattened sequence length.
- $h$: number of attention heads.
- $N_q$: number of object queries.
- $N_{gt}$: number of GT objects in an image.
- $K$: number of foreground classes.
- logits shape: $(B, N_q, K+1)$.
- boxes shape: $(B, N_q, 4)$ with normalized $(c_x,c_y,w,h)$.

This theory file is intended to be used together with the pipeline document in [researches/overview_pipeline.md](researches/overview_pipeline.md).