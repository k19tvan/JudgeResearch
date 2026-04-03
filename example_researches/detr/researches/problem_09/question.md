<!-- learning_path/problem_09/question.md -->
# Problem 09 Questions

## Multiple Choice
1. How does the Hungarian matcher compare to Anchor-based systems used in YOLO or Faster-RCNN?
A. It provides multi-target overlaps for every query organically
B. It entirely replaces IoU-based anchor heuristic matching with pure optimal bipartite 1-to-1 matching
C. It operates only on cropped ROI boxes
D. It performs identical logic but only runs on the CPU

2. Why must we use `scipy.optimize.linear_sum_assignment` on the CPU?
A. Because PyTorch GPU has no mathematically equivalent function in its default library to solve combinatorial min-cost bipartite graphs
B. Because it is inherently parallelizable on CPU
C. To save GPU memory strictly
D. It's an arbitrary preference 

## Answer Key
1.B 2.A
