# Problem 01 Questions

## Multiple Choice

1. Why do object detectors often predict bounding boxes in (CX, CY, W, H) format instead of (X1, Y1, X2, Y2)?
A. Because neural networks cannot generate negative numbers.
B. It is computationally cheaper to calculate gradients for centers and dimensions.
C. Normalizing center coordinates and regressing logarithmic dimensions usually stabilizes training gradients.
D. PyTorch only supports the (CX, CY, W, H) format natively.

2. In PyTorch, what does `tensor.unbind(-1)` do?
A. Removes the tensor from GPU memory.
B. Flattens the entire tensor into a 1D vector.
C. Slices the tensor along the last dimension, returning a tuple of all slices.
D. Unfreezes the tensor's gradients.

3. If you input a tensor of shape (32, 100, 4) representing 32 frames of 100 bounding boxes, what shape will `x_1` be after unbinding?
A. (4,)
B. (32, 100)
C. (32, 4)
D. (100, 4)

4. Why must we avoid Python `for` loops when converting thousands of bounding box coordinates?
A. Python loops randomly shuffle tensor memory.
B. Python loops bypass the Global Interpreter Lock (GIL).
C. They prevent the underlying C++/CUDA backend from parallelizing the mathematical operations.
D. They convert float32 to float16 implicitly.

5. What would happen if width (`w`) or height (`h`) are negative values during conversion?
A. PyTorch raises a compilation error naturally.
B. Area formulas later in the pipeline will calculate incorrect intersections and gradients will explode.
C. The box format automatically corrects itself and takes the absolute value.
D. It simply skips calculating those boxes with zero penalty.

## Answer Key
1.C 2.C 3.B 4.C 5.B
