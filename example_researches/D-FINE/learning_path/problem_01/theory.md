# Problem 01 Theory - Bounding Box Coordinate Conversions

## Core Definitions
Bounding boxes pinpoint the location of objects. D-FINE uses normalized center coordinates to predict offsets easily via the decoder, but absolute corner coordinates to evaluate overlap (IoU) with the ground truth targets during the Hungarian Matching phase.

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| `x` | `(..., 4)` | Input bounding box tensor. The `...` represents any number of batch or sequence dimensions (e.g., `(B, T, 4)` for a batch). The last dimension `D=4` holds the box coordinates. |
| `cx, cy` | `(..., 1)` | Center X and Y coordinates. |
| `w, h` | `(..., 1)` | Box width and height. |
| `x1, y1` | `(..., 1)` | Top-left corner coordinates. |
| `x2, y2` | `(..., 1)` | Bottom-right corner coordinates. |

## Main Equations (LaTeX)

To convert from Center-Width-Height ($cx, cy, w, h$) to Corners ($x_1, y_1, x_2, y_2$):
$$ x_1 = cx - \frac{w}{2} $$
$$ y_1 = cy - \frac{h}{2} $$
$$ x_2 = cx + \frac{w}{2} $$
$$ y_2 = cy + \frac{h}{2} $$

To reverse the conversion from Corners to Center-Width-Height:
$$ w = x_2 - x_1 $$
$$ h = y_2 - y_1 $$
$$ cx = x_1 + \frac{w}{2} $$
$$ cy = y_1 + \frac{h}{2} $$

## Step-by-Step Derivation or Computation Flow
1. **Unbinding**: Given a tensor of shape `(..., 4)`, slice or unbind it along the last dimension to extract the 4 individual components.
2. **Arithmetic**: Apply the linear shifts (dividing width/height by 2) to compute the new coordinates.
3. **Stacking**: Concatenate or stack the 4 computed tensors back together along the last dimension to reform a `(..., 4)` tensor.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- `Input`: `x` shaped `(B, 4)`
- `Unbind`: `x.unbind(-1)` yields 4 tensors `cx`, `cy`, `w`, `h`, each shaped `(B,)`
- `Compute`: `b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]` creates a list of 4 tensors of shape `(B,)`
- `Output`: `torch.stack(b, dim=-1)` yields a tensor of shape `(B, 4)`

## Practical Interpretation
By vectorizing mathematical unbinds and stacks, this operation runs entirely in C++/CUDA under the PyTorch hood. In a batch of 32 images with 300 box predictions each, `(32, 300, 4)`, these matrix operations will update 9,600 boxes instantly without any Python loop iteration overhead.
