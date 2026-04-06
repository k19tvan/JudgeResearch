# Problem 02 Questions

## Multiple Choice

1. What shape results from broadcasting `boxes1[:, None, :]` and `boxes2[None, :, :]` across `N` and `M` pairs of boxes?
A. `(N, M, 4)`
B. `(N+M, 4)`
C. `(4, N, M)`
D. `(N*M, 4)`

2. Why do we clamp the intersection width/height to a minimum of 0.0 using `torch.clamp(.., min=0.0)`?
A. Because neural networks cannot output negative numbers.
B. To prevent disjoint boxes from contributing negative areas to the intersection sum, which ruins the union calculation.
C. It acts as an activation function (ReLU) for the output layer.
D. To prevent divide-by-zero errors in hardware.

3. Standard IoU is highly bounded. What happens to standard IoU when two boxes are far apart?
A. It approaches -1.
B. It equals 0, and the gradient vanishes, leaving the model confused about which direction to optimize.
C. It equals exactly 1.
D. It grows infinitely large (nan).

4. Which part of the GIoU equation produces the scale-invariant penalty?
A. `Area(Intersection) / Area(Union)`
B. `(Area(C) - Area(Union)) / Area(C)`
C. `Area(C) - Area(Intersection)`
D. `Area(Intersection) + Area(Union)`

5. If a predicted box perfectly encloses and matches the truth box, what is the value of both its IoU and GIoU?
A. 0.0
B. 0.5
C. 1.0
D. 100.0

## Answer Key
1.A 2.B 3.B 4.B 5.C