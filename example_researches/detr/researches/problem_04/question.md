<!-- learning_path/problem_04/question.md -->
# Problem 04 Questions

## Multiple Choice
1. Why is positional embedding added to the Queries and Keys but NOT to the Values?
A. To reduce memory overload
B. The Softmax function rejects embedded Values
C. The network should retrieve original unmodified image features (Values) but needs position constraints to know where to pull them from (Q, K).
D. Values are inherently 1-dimensional

2. What is the standard scaling factor applied before the Softmax operation?
A. $\sqrt{B}$
B. $\sqrt{d_k}$
C. $1.0$
D. Natural Logarithm 

## Answer Key
1.C 2.B
