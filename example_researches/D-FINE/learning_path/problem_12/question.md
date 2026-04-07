# Problem 12 Questions

## Multiple Choice

1. Why is `optimizer.zero_grad()` called BEFORE the forward pass, not after?
   A. To clear gradients from the previous batch, preventing gradient accumulation across batches
   B. Forward pass requires zero gradients to initialize
   C. To reset the BatchNorm statistics
   D. It's only called before the backward pass

2. In D-FINE's training, why does the Hungarian Matching change each epoch?
   A. The matcher is randomly initialized each time
   B. As model parameters update, predictions change, so the optimal matching between predictions and GTs shifts
   C. The matcher filters predictions by confidence threshold
   D. Matching is fixed after epoch 1

3. What is the key behavioral difference between `model.train()` and `model.eval()` modes in D-FINE?
   A. Different forward pass topology
   B. Dropout is active in train (adds noise) and disabled in eval; BatchNorm uses batch stats in train and running stats in eval
   C. Different number of decoder layers
   D. The loss function is only computed in train mode

4. The AdamW optimizer's "decoupled weight decay" means:
   A. Weight decay is applied to all layers except the last
   B. L2 regularization is applied directly to weights (not through gradients), preventing adaptive learning rate from scaling the regularization
   C. Weight decay decays over time independently
   D. Weight decay is computed on a separate GPU

5. After 10 training iterations with random data and random model initialization, what should we expect about the detection quality?
   A. Near-perfect detection (IoU ≈ 1.0) since the model is simple
   B. Poor but finite IoU (typically < 0.1), as the model is still random — the purpose is verifying the pipeline runs, not performance
   C. Exactly 0.0 IoU since no real data was used
   D. IoU ≥ 0.5 due to the DETR inductive bias

## Answer Key
1.A 2.B 3.B 4.B 5.B
