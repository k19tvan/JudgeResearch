#### **Problem Description**
In object detection models (such as DETR), evaluating the deviation between bounding boxes is a fundamental task. The traditional **Intersection over Union (IoU)** metric has a major drawback: when two bounding boxes do not overlap (IoU = 0), the gradient becomes 0. This "zero-gradient" issue prevents the model from learning and optimizing the distance between non-overlapping boxes during training.

To address this limitation, the **Generalized Intersection over Union (GIoU)** metric was introduced. GIoU not only evaluates the overlapping area but also penalizes boxes that are far apart or mismatched in scale and position, even when they have no overlap at all.

Given $N$ predicted bounding boxes and $M$ ground truth bounding boxes, where each box is represented by 4 absolute coordinates $[x_{min}, y_{min}, x_{max}, y_{max}]$, your task is to compute the $N \times M$ GIoU matrix. The element at row $i$ and column $j$ should represent the GIoU value between the $i$-th predicted box and the $j$-th ground truth box.

**GIoU Formula for Two Boxes $A$ and $B$:**
1. Compute the **Intersection** area: $I = \text{Area}(A \cap B)$
2. Compute the **Union** area: $U = \text{Area}(A \cup B)$
3. Calculate the standard **IoU**: $\text{IoU} = \frac{I}{U}$
4. Find the **smallest enclosing box** $C$ that covers both $A$ and $B$.
5. Compute **GIoU**: 
   $$\text{GIoU} = \text{IoU} - \frac{\text{Area}(C) - U}{\text{Area}(C)}$$

#### **Input Specification**
The input is provided via standard input (stdin) as a single **JSON object** with the following keys:
- `"pred_boxes"`: A 2D JSON array of size $N \times 4$ representing the coordinates of the predicted bounding boxes. Each box is represented as `[x_min, y_min, x_max, y_max]`.
- `"gt_boxes"`: A 2D JSON array of size $M \times 4$ representing the coordinates of the ground truth bounding boxes. Each box is represented as `[x_min, y_min, x_max, y_max]`.

*Constraints:*
- $1 \le N, M \le 1000$
- $0 \le x_{min} < x_{max} \le 10^4$ and $0 \le y_{min} < y_{max} \le 10^4$.
- The floating-point numbers in the input JSON have at most 2 decimal places.

#### **Output Specification**
The program must output to standard output (stdout) a single **JSON object** with the following key:
- `"output"`: A 2D JSON array (matrix) of size $N \times M$, where the element at `output[i][j]` represents the computed GIoU value between the $i$-th predicted box (`pred_boxes[i]`) and the $j$-th ground truth box (`gt_boxes[j]`).

*Precision Constraint:*
- Your output will be considered correct if the absolute or relative error for each GIoU value in the `"output"` matrix does not exceed $10^{-6}$.

#### **Example Input (JSON)**
```json
{
    "pred_boxes": [
      [0.0, 0.0, 2.0, 2.0],
      [1.0, 1.0, 3.0, 3.0]
    ],
    "gt_boxes": [
      [0.0, 0.0, 2.0, 2.0],
      [1.0, 1.0, 3.0, 3.0]
    ]
}