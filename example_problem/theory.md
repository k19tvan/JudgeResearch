#### **Theoretical Background: IoU vs. GIoU**
In classical Object Detection evaluation, **Intersection over Union (IoU)** is the standard metric defined as:
$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

##### **The Gradient Vanishing Limitation**
When a model is being trained and the predicted bounding box $A$ does not overlap with the ground truth box $B$ ($A \cap B = \emptyset$), then $\text{IoU}(A, B) = 0$. 

If we define our localization loss function purely based on IoU as $\mathcal{L}_{IoU} = 1 - \text{IoU}$, the loss value remains constant at $1.0$ for any non-overlapping positions. Consequently, the gradient of the loss with respect to the coordinates is:
$$\nabla \mathcal{L}_{IoU} = 0$$

Without a non-zero gradient, backpropagation cannot guide the predicted box toward the ground truth box.

##### **The GIoU Solution**
Generalized Intersection over Union (GIoU) introduces a penalty term utilizing the **smallest enclosing box** $C$ (or the convex hull of $A \cup B$):
$$\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{\text{Area}(C \setminus (A \cup B))}{\text{Area}(C)} = \text{IoU} - \frac{\text{Area}(C) - \text{Area}(A \cup B)}{\text{Area}(C)}$$

Where:
- $C$ is the smallest rectangle that contains both $A$ and $B$.
- $\text{Area}(C \setminus (A \cup B))$ represents the empty space inside $C$ that is not covered by either $A$ or $B$.

##### **Properties of GIoU:**
1. **Symmetric:** $\text{GIoU}(A, B) = \text{GIoU}(B, A)$.
2. **Scale Invariant:** Like IoU, it is independent of the scale of the bounding boxes.
3. **Bounded Range:** $-1 \le \text{GIoU}(A, B) \le 1$.
   - $\text{GIoU} = 1$ when the two boxes align perfectly.
   - $\text{GIoU} \to -1$ as the distance between the non-overlapping boxes approaches infinity.
4. **Continuous Gradient:** Even when the overlap is zero, moving the boxes closer together reduces $\text{Area}(C)$, which increases $\text{GIoU}$ (and decreases $\mathcal{L}_{GIoU} = 1 - \text{GIoU}$), providing a continuous gradient for optimization.
