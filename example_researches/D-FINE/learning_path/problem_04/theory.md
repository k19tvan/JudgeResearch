# Problem 04 Theory - Varifocal Loss (VFL) for Classification

## Core Definitions
- **Focal Loss**: Modifies BCE by downweighting easy negatives via `(1-p_t)^gamma`, focusing training on hard examples.
- **Varifocal Loss (VFL)**: Extends focal loss for quality estimation. Instead of binary {0/1} targets, VFL uses IoU between prediction and ground truth as the positive target score — directly modeling "quality" of a prediction.
- **Why VFL in D-FINE**: DETR-based detectors set only a few queries as positives. Using IoU as the quality score makes the confidence score a true indicator of localization quality, enabling better evaluation without NMS calibration tricks.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `pred_logit` | `(N, C)` | Raw class logits for N matched predictions |
| `p` | `(N, C)` | Sigmoid probabilities |
| `gt_score` | `(N,)` | IoU quality of each positive prediction |
| `label` | `(N,)` | Class index for each positive |
| `one_hot` | `(N, C)` | One-hot encoding of class labels |
| `target` | `(N, C)` | Soft target: `gt_score` at gt class, 0 elsewhere |
| `weight` | `(N, C)` | Sample weight for focal modulation |
| `loss` | `(N, C)` | Per-element VFL loss before reduction |

## Main Equations (LaTeX)

**Sigmoid Focal Loss (standard):**
$$ p_t = p \cdot t + (1-p) \cdot (1-t), \quad p = \sigma(\text{logit}), \quad t \in \{0,1\} $$
$$ \mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$

**Varifocal Loss target (soft quality label):**
$$ q_{ij} = \begin{cases} \text{IoU}(\hat{b}_i, b_j) & \text{if } j = \text{class}(i) \\ 0 & \text{otherwise} \end{cases} $$

**VFL weight:**
$$ w_{ij} = \begin{cases} q_{ij} & \text{if positive (matched)} \\ \alpha \cdot p_{ij}^\gamma & \text{if negative (background)} \end{cases} $$

**VFL loss element:**
$$ \mathcal{L}_{VFL} = w_{ij} \cdot \text{BCE}(\text{logit}_{ij},\ q_{ij}) $$

## Step-by-Step Derivation or Computation Flow
1. Compute `p = sigmoid(pred_logit)` → `(N, C)`.
2. Build `one_hot = F.one_hot(label, C).float()` → `(N, C)`.
3. Build soft target: `target = one_hot * gt_score.unsqueeze(-1)` → `(N, C)`.
4. Build weight: `weight = alpha * p.pow(gamma) * (1 - one_hot) + one_hot * gt_score.unsqueeze(-1)`.
5. Compute: `loss = F.binary_cross_entropy_with_logits(pred_logit, target, weight=weight, reduction='none')`.
6. Return `loss` of shape `(N, C)`.

## Tensor Shape Flow
```
pred_logit: (N, C) → sigmoid → p: (N, C)
label:      (N,)   → one_hot → (N, C)
gt_score:   (N,)   → unsqueeze(-1) → (N, 1)
                         ↓
target:  (N, C)   weight: (N, C)
                         ↓
BCE with weight → loss: (N, C)
```

## Practical Interpretation
In D-FINE's `loss_labels_vfl`, the IoU of the matched predicted box with the ground truth is used as `gt_score`. A prediction that localizes perfectly (IoU→1) gets a target score of ~1.0; a sloppily localized prediction gets a lower target. This forces the model to calibrate its class confidence proportionally to box quality — a property used directly during NMS-free inference to rank candidates.

**Mini-example (N=1, C=3):**
- Prediction class=0, IoU=0.8. label=[0], gt_score=[0.8].
- target = [0.8, 0, 0]. weight_pos = 0.8, weight_neg = 0.25*p²_neg.
- Loss = 0.8 * BCE(logit_0, 0.8) + 0.25*p₁²*BCE(logit_1, 0) + 0.25*p₂²*BCE(logit_2, 0).
