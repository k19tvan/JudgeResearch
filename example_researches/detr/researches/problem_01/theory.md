# Problem 01 Theory - Box Format Conversions

## Core Definitions
- Bounding boxes can be represented as absolute corner form $(x_{min}, y_{min}, x_{max}, y_{max})$, absolute COCO form $(x_{min}, y_{min}, w, h)$, or normalized center form $(c_x, c_y, w_n, h_n)$.
- DETR predicts normalized center form so predictions are resolution-invariant.

## Variables and Shape Dictionary
- $B$: number of boxes in a batch item.
- $W_{img}, H_{img}$: image width and height in pixels.
- $x_{min}, y_{min}$: top-left corner in pixels.
- $w, h$: width and height in pixels.
- $c_x, c_y$: box center coordinates.
- Absolute input shape: $(B, 4)$.
- Normalized output shape: $(B, 4)$.

## Main Equations (LaTeX)
$$
c_x = \frac{x_{min} + \frac{w}{2}}{W_{img}}, \quad c_y = \frac{y_{min} + \frac{h}{2}}{H_{img}}
$$
$$
w_n = \frac{w}{W_{img}}, \quad h_n = \frac{h}{H_{img}}
$$
$$
x_{min} = (c_x - \frac{w_n}{2})W_{img}, \; y_{min} = (c_y - \frac{h_n}{2})H_{img}, \; x_{max} = (c_x + \frac{w_n}{2})W_{img}, \; y_{max} = (c_y + \frac{h_n}{2})H_{img}
$$

## Step-by-Step Derivation or Computation Flow
1. Read absolute COCO tuple $(x_{min}, y_{min}, w, h)$.
2. Compute absolute center coordinates.
3. Divide center and size by image dimensions to normalize into $[0,1]$.
4. For inverse transform, rescale normalized values to pixels and recover corners.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input: absolute xywh tensor $(B, 4)$.
- Intermediate: center/size absolute tensor $(B, 4)$.
- Output A: normalized cxcywh tensor $(B, 4)$.
- Output B: absolute xyxy tensor $(B, 4)$.

Worked mini-example:
- Let $B=2$, $W_{img}=640$, $H_{img}=480$.
- Box 1: $(64, 48, 128, 96)$ gives $c_x=(64+64)/640=0.2$, $c_y=(48+48)/480=0.2$, $w_n=0.2$, $h_n=0.2$.

## Practical Interpretation
- Normalized representation stabilizes learning across varying resolutions.
- Corner representation is convenient for IoU/GIoU and visualization.
- Correct conversion is foundational; small coordinate bugs propagate to matcher and losses.