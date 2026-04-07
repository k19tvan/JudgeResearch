# Problem 02 Theory - Generalized Intersection over Union (GIoU)

## Core Definitions
- **IoU (Intersection over Union)**: The ratio of the area of intersection relative to the area of union of two bounding boxes. It is the standard spatial overlap metric.
- **GIoU**: Extends IoU by adding a penalty term that accounts for the area of the minimum enclosing box that is NOT covered by either box. This ensures that even non-overlapping boxes return a meaningful gradient.
- **In D-FINE**: GIoU appears in (1) the Hungarian Matcher cost matrix as `cost_giou = -GIoU(pred, gt)` and (2) in `SetCriterion.loss_boxes` as `loss_giou = 1 - diag(GIoU)`.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `boxes1` | `(N, 4)` | N predicted boxes in `xyxy` format |
| `boxes2` | `(M, 4)` | M ground truth boxes in `xyxy` format |
| `lt` | `(N, M, 2)` | Intersection top-left corner (element-wise max) |
| `rb` | `(N, M, 2)` | Intersection bottom-right corner (element-wise min) |
| `inter` | `(N, M)` | Intersection area |
| `union` | `(N, M)` | Union area |
| `iou` | `(N, M)` | Standard IoU matrix |
| `encl_lt` | `(N, M, 2)` | Enclosing box top-left (element-wise min) |
| `encl_rb` | `(N, M, 2)` | Enclosing box bottom-right (element-wise max) |
| `encl_area` | `(N, M)` | Area of minimum enclosing box |
| `giou` | `(N, M)` | GIoU matrix in `[-1, 1]` |

## Main Equations (LaTeX)

**Area of each box:**
$$ \text{area}(b) = (x_2 - x_1) \cdot (y_2 - y_1) $$

**Intersection area:**
$$ \text{inter} = \max(0,\ \min(x_2^A, x_2^B) - \max(x_1^A, x_1^B)) \cdot \max(0,\ \min(y_2^A, y_2^B) - \max(y_1^A, y_1^B)) $$

**Union area:**
$$ \text{union} = \text{area}(A) + \text{area}(B) - \text{inter} $$

**Standard IoU:**
$$ \text{IoU}(A, B) = \frac{\text{inter}}{\text{union}} $$

**GIoU:**
$$ \text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|\text{encl}(A,B)| - \text{union}}{|\text{encl}(A,B)|} $$

where $|\text{encl}(A,B)|$ is the area of the smallest axis-aligned bounding box enclosing both $A$ and $B$.

## Step-by-Step Derivation or Computation Flow
1. Compute `area1 = (x2-x1)*(y2-y1)` for each box in `boxes1`, giving shape `(N,)`.
2. Compute `area2`, giving `(M,)`.
3. Broadcast: `lt = max(boxes1[:,None,:2], boxes2[:,:2])` → `(N,M,2)`.
4. `rb = min(boxes1[:,None,2:], boxes2[:,2:])` → `(N,M,2)`.
5. `wh = clamp(rb - lt, min=0)` → `(N,M,2)`. Multiply channels: `inter = wh[...,0] * wh[...,1]` → `(N,M)`.
6. `union = area1[:,None] + area2 - inter` → `(N,M)`.
7. `iou = inter / union` → `(N,M)`.
8. Enclosing: `encl_lt = min(boxes1[:,None,:2], boxes2[:,:2])`, `encl_rb = max(...)`.
9. `encl_wh = clamp(encl_rb - encl_lt, min=0)` → `(N,M,2)`.
10. `encl_area = encl_wh[...,0] * encl_wh[...,1]` → `(N,M)`.
11. `giou = iou - (encl_area - union) / encl_area`.

## Tensor Shape Flow (Input → Intermediate → Output)
```
boxes1: (N, 4)
boxes2: (M, 4)
  ↓ broadcast
lt, rb: (N, M, 2)
  ↓ clamp → multiply channels
inter: (N, M)
union: (N, M)
  ↓
iou: (N, M)
  ↓ + enclosing penalty
giou: (N, M)
```

## Practical Interpretation
When two predicted boxes don't overlap with any ground truth box, their standard IoU is identically 0 — no gradient flows to improve them. GIoU provides a negative signal (proportional to how far boxes are from each other) that pushes predictions toward ground truth, dramatically accelerating DETR training convergence.

**Mini-example (N=1, M=1):**
- A = [0, 0, 2, 2], area=4. B = [3, 3, 5, 5], area=4.
- Intersection = 0. Union = 8. IoU = 0.
- Enclosing = [0,0,5,5], area=25.
- GIoU = 0 − (25−8)/25 = −0.68. ← meaningful negative signal!
