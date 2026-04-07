# Learning Path Overview

This structured path decompiles the D-FINE repository into exactly 12 small, verifiable problems, strictly ordered by mathematical dependencies.

| ID | Problem Name | Theory Goal | Coding Goal | Depends On | Repo Module Mapped | End-to-End Phase |
|---|---|---|---|---|---|---|
| **01** | Bounding Box Basics | Understand normalized `cxcywh` vs corner `xyxy` box representations | Implement robust box conversion utilities | None | `src/core/box_ops.py` | Theory & Math |
| **02** | GIoU Computation | Understand Generalized Intersection over Union metrics | Implement vectorized IoU and GIoU tensor calculation | P01 | `src/core/box_ops.py` | Theory & Math |
| **03** | Hungarian Matcher | Understand bipartite matching for set-based prediction | Implement bounding box and classification cost matrix | P01, P02 | `src/core/matcher.py` | Theory & Math |
| **04** | Focal Loss | Understand mitigating class imbalances safely | Formulate focal loss with `alpha` and `gamma` | None | `src/core/loss.py` | Theory & Math |
| **05** | DETR Set Criterion | Master aggregation of classification and bounding box losses | Wire matcher and metrics into a unified `SetCriterion` module | P02, P03, P04 | `src/models/dfine/criterion.py` | Training Components |
| **06** | Sine 2D PE | Grasp spatial bias injections in Transformers | Build a static 2D Sine/Cosine Positional Encoding generator | None | `src/nn/pe.py` | Model Architecture |
| **07** | Multi-Head Attention | Solidify Queries, Keys, Values and Scaled Dot-Product logic | Implement pure PyTorch standard Multi-Head Attention | None | `src/nn/transformer.py` | Model Architecture |
| **08** | Transformer Encoder | Understand self-attention over flattened feature maps | Implement one robust Encoder block with feedforward | P06, P07 | `src/nn/transformer.py`| Model Architecture |
| **09** | Transformer Decoder | Understand object queries fetching visual properties | Implement a Decoder layer using Cross-Attention to image | P06, P07 | `src/models/dfine/decoder.py` | Model Architecture |
| **10** | HGNetV2-S Stem | Parse modern efficient CNN feature extractors | Build the Conv Stem for hierarchical patch extraction | None | `src/nn/backbone/hgnetv2.py`| Model Architecture |
| **11** | Model Assembly | Master the data flow traversing Core + Encoder + Decoder + Head | Link modules to form the complete `DFINE` forward pass | P08, P09, P10 | `src/models/dfine/dfine.py` | Capstone |
| **12** | Train & Eval Loop | Run the holistic learning pipeline | Implement the optimization loop with the `SetCriterion` | P05, P11 | `src/solver/`, `train.py` | Capstone |
