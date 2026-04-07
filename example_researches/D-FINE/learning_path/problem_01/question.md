# Problem 01 Questions

## Multiple Choice
1. Why does DETR prioritize modeling using `cxcywh` natively instead of static `xyxy` predictions?
   A. Native Cartesian rendering operates quicker
   B. Prevents negative coordinates during calculation
   C. Predicts symmetric point anchors scaling around features easily without edge biasing
   D. `xyxy` enforces float64 memory allocations out of boundaries
   
2. Which PyTorch operation successfully isolates the final bounding box coordinate axis without maintaining a rigid dimension size of 1?
   A. `torch.split`
   B. `torch.unbind`
   C. `torch.index_select`
   D. `torch.chunk`
   
3. What mapping denotes the holistic image frame when bounded correctly in `cxcywh` assuming a normalized format `[0, 1]`?
   A. `[1.0, 1.0, 0.5, 0.5]`
   B. `[0.0, 0.0, 1.0, 1.0]`
   C. `[0.5, 0.5, 1.0, 1.0]`
   D. `[1.0, 1.0, 1.0, 1.0]`
   
4. How do bounding architectures structurally safeguard neural constraints on elements natively requiring positive spans (width, height)?
   A. Through exponential wrappers / Activation logic downstream before regression loss
   B. Utilizing Softmax layers
   C. Zero normalization algorithms 
   D. It’s naturally unconstrained
   
5. When synthesizing the isolated tensors into a unified output matrix, what stack parameter securely packs them back logically inside the trailing dimensional graph?
   A. `dim=None`
   B. `dim=0`
   C. `dim=1`
   D. `dim=-1`

## Answer Key
1.C 2.B 3.C 4.A 5.D
