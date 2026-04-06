# Problem 02 Theory - Generalized IoU (GIoU)

## Core Definitions
While absolute coordinate losses (like L1) are useful, they do not perfectly correlate with human perception of bounding box quality. IoU is scale-invariant. GIoU adds a penalty when boxes are disjoint, which prevents vanishing gradients.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| `boxes1` | `(N, 4)` | First set of boxes (e.g., ground truth). |
| `boxes2` | `(M, 4)` | Second set of boxes (e.g., predictions). |
| `area1`, `area2` | `(N,)`, `(M,)` | Surface area of each box. |
| `inter` | `(N, M)` | Pairwise intersection area. |
| `union` | `(N, M)` | Pairwise union area. |
| `iou` | `(N, M)` | Pairwise Intersection over Union. |
| `enclosing_area` | `(N, M)` | Area of the smallest convex hull enclosing both boxes. |

## Main Equations (LaTeX)

Standard IoU is the ratio of Intersection area over Union area:
$$ IoU = \frac{Area(A \cap B)}{Area(A \cup B)} $$

Union is calculated as the sum of individual areas minus their intersection:
$$ Area(A \cup B) = Area(A) + Area(B) - Area(A \cap B) $$

Generalized IoU adds a penalty based on the smallest enclosing box $C$:
$$ GIoU = IoU - \frac{Area(C) - Area(A \cup B)}{Area(C)} $$

## Step-by-Step Derivation or Computation Flow
1. Compute areas of all `boxes1` and `boxes2`.
2. Compute pairwise intersections: find `max(x1)` and `max(y1)` for top-left, `min(x2)` and `min(y2)` for bottom-right. Clamp coordinates to avoid negative areas.
3. Compute `union = area1 + area2 - inter`.
4. Calculate `iou = inter / union`.
5. Compute the enclosing box $C$ by finding `min(x1, y1)` and `max(x2, y2)`.
6. Calculate `area_c`.
7. Compute `giou = iou - (area_c - union) / area_c`.

## Practical Interpretation
If a network proposes a box completely outside the image where the ground truth is, standard IoU is 0, giving no direction to move. GIoU will gracefully decrease (approaching -1) the further away the predicted box moves, thus giving a smooth gradient path back towards the target object.