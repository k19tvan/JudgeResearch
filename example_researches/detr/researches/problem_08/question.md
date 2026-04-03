<!-- learning_path/problem_08/question.md -->
# Problem 08 Questions

## Multiple Choice
1. Why do we disable gradients when computing the Bipartite Cost Matrix?
A. Because we just use it for assigning labels, not training parameters directly
B. Computing gradients for an $N \times M$ matrix is mathematically impossible
C. SciPy linear assignment supports backward hooks anyway
D. It prevents catastrophic forgetting

2. Does the model output `N` predictions for an image that only has `M=3` targets?
A. Yes, it predicts exactly `N` independent bounding boxes regardless of input density
B. No, `N` scales dynamically
C. No, it stops decoding after the first `M`

## Answer Key
1.A 2.A
