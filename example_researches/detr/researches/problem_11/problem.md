<!-- learning_path/problem_11/problem.md -->
# Problem 11 - Capstone: End-To-End Training Step

## Description
- Assemble everything! 
- Given input images and targets, compute the forward pass, find assignments, apply Loss, and run the optimizer step.

## Requirements
- Use `optimizer.zero_grad(set_to_none=True)` for efficiency.
- Forward model.
- Pair using matcher.
- Gather loss from criterion.
- Backward step `loss.backward()`.
- Explicit gradient clipping via `torch.nn.utils.clip_grad_norm_`.
- `optimizer.step()`

## Theory Snapshot
- A training step composes forward pass, assignment, loss computation, gradient propagation, and parameter update.
- The loop semantics ensure gradients flow only through differentiable parts while matching is non-differentiable.
- Full computational graph explanation is in [researches/problem_11/theory.md](researches/problem_11/theory.md).

## Checker
```bash
python learning_path/problem_11/checker.py
```
