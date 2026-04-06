# Problem 02 Theory - Generalized IoU (GIoU)

## Core Definitions
- IoU measures overlap between two boxes.
- GIoU extends IoU by penalizing separation when boxes do not overlap.

## Variables and Shape Dictionary
- $A, B$: two boxes in corner form $(x_{min}, y_{min}, x_{max}, y_{max})$.
- $|\cdot|$: area operator.
- $C$: smallest enclosing box covering both $A$ and $B$.
- Pred boxes shape: $(N, 4)$.
- Target boxes shape: $(M, 4)$.
- Pairwise IoU/GIoU output shape: $(N, M)$.

## Main Equations (LaTeX)
$$
\text{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$
$$
|A\cup B|=|A|+|B|-|A\cap B|
$$
$$
\text{GIoU}(A,B)=\text{IoU}(A,B)-\frac{|C\setminus(A\cup B)|}{|C|}
$$

## Step-by-Step Derivation or Computation Flow
1. Compute intersection width/height via clamped corner overlaps.
2. Compute intersection area and union area.
3. Compute IoU as intersection over union.
4. Build enclosing box $C$ from min top-left and max bottom-right corners.
5. Subtract enclosure penalty from IoU to obtain GIoU.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input: pred $(N,4)$, target $(M,4)$.
- Broadcasted intersections: $(N,M)$ widths/heights.
- IoU matrix: $(N,M)$.
- Enclosure penalty matrix: $(N,M)$.
- Output GIoU matrix: $(N,M)$.

Worked mini-example:
- If $N=2, M=3$, output has shape $(2,3)$.
- Each entry $(i,j)$ is the GIoU between predicted box $i$ and target box $j$.

## Practical Interpretation
- IoU alone gives zero gradient signal for non-overlapping boxes.
- GIoU keeps training informative by encoding how far boxes are apart spatially.
- DETR uses GIoU in both matcher cost and loss.