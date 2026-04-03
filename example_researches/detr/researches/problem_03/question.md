<!-- learning_path/problem_03/question.md -->
# Problem 03 Questions

## Multiple Choice
1. How does the position embedding handle padded regions in the image?
A. It throws an error
B. It assigns them a 0 representation, halting the cumulative position counter
C. It treats them exactly like normal pixels
D. It drops them from the memory map

2. Why is `num_pos_feats` set to half of the Transformer hidden dimension?
A. Because we concatenate the separately generated X and Y encodings
B. To save parameters
C. It allows alternating between Sine and Cosine
D. Because the model merges RGB channels first

## Answer Key
1.B 2.A
