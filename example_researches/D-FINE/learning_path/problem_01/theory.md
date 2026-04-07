# Problem 01 Theory - Bounding Box Format Conversion

## Core Definitions
- **`cxcywh`**: Center-based format native to the neural network’s terminal regression head. Predicts center offsets and exponential relative sizing. In DETR configurations these elements are characteristically constrained within normalized `[0, 1]` ranges.
- **`xyxy`**: Cartesian coordinate pairs representing the Top-Left and Bottom-Right bounding anchor edges. Geometric assessments natively require these to process spatial area and intersection maps.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
|---|---|---|
| `boxes` | `(..., 4)` | Input bounding box tensor, dynamic leading dimensions. |
| `c_x, c_y` | `(...)` | Center anchor coordinates. |
| `w, h` | `(...)` | Object width and heights. |
| `x_1, y_1` | `(...)` | Top-Left X and Y bounds. |
| `x_2, y_2` | `(...)` | Bottom-Right X and Y bounds. |

## Main Equations (LaTeX)

**Targeting `xyxy` from `cxcywh`:**
$$ x_1 = c_x - \frac{w}{2} $$
$$ y_1 = c_y - \frac{h}{2} $$
$$ x_2 = c_x + \frac{w}{2} $$
$$ y_2 = c_y + \frac{h}{2} $$

**Re-Targeting `cxcywh` from `xyxy`:**
$$ c_x = \frac{x_1 + x_2}{2} $$
$$ c_y = \frac{y_1 + y_2}{2} $$
$$ w = x_2 - x_1 $$
$$ h = y_2 - y_1 $$

## Step-by-Step Derivation or Computation Flow
1. Consume the `boxes` tensor. 
2. Initiate `unbind` targeting the `-1` axis, producing 4 explicit sub-tensors mirroring the initial multi-dimensional matrix.
3. Calculate new offsets scaling values by `.5` for symmetric distribution from the central hub, mapping them against axis positions. 
4. Pack these mathematically transformed sub-tensors linearly into a discrete list.
5. Command `torch.stack()` to reconnect the axes against `dim=-1`.

## Practical Interpretation
Detectors functionally perform superiorly outputting local structural offsets versus arbitrary pixel thresholds globally. However, structural mathematical tasks downstream (specifically GIoU checks and Set Match penalties) physically cannot work with abstract center offsets; they demand pure geometrical collision thresholds. Vectorizing this transformation prevents critical bottleneck logic inside loops per predicted box frame during loss metric derivations.
