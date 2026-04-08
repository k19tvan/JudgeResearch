# Learning Path Overview

This structured path decompiles the D-FINE repository into 15 small, verifiable problems, strictly ordered by mathematical dependencies.

**📊 Architecture Diagrams**: See [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) and [MERMAID_FLOWCHARTS.md](MERMAID_FLOWCHARTS.md) for complete visual references.

## Phase 1: Foundation (Problems 01–05)
Core geometric and loss computation building blocks.

| ID | Problem Name | Theory Goal | Coding Goal | Depends On | Repo Module Mapped | End-to-End Phase |
|---|---|---|---|---|---|---|
| **01** | Bounding Box Basics | Understand normalized `cxcywh` vs corner `xyxy` box representations | Implement robust box conversion utilities | None | `src/core/box_ops.py` | Theory & Math |
| **02** | GIoU Computation | Understand Generalized Intersection over Union metrics | Implement vectorized IoU and GIoU tensor calculation | P01 | `src/core/box_ops.py` | Theory & Math |
| **03** | Hungarian Matcher | Understand bipartite matching for set-based prediction | Implement bounding box and classification cost matrix | P01, P02 | `src/core/matcher.py` | Theory & Math |
| **04** | Focal Loss | Understand mitigating class imbalances safely | Formulate focal loss with `alpha` and `gamma` | None | `src/core/loss.py` | Theory & Math |
| **05** | DETR Set Criterion | Master aggregation of classification and bounding box losses | Wire matcher and metrics into a unified `SetCriterion` module | P02, P03, P04 | `src/models/dfine/criterion.py` | Training Components |

## Phase 2: Architecture (Problems 06–12)
Model architecture components and training pipeline.

| ID | Problem Name | Theory Goal | Coding Goal | Depends On | Repo Module Mapped | End-to-End Phase |
|---|---|---|---|---|---|---|
| **06** | Sine 2D PE | Grasp spatial bias injections in Transformers | Build a static 2D Sine/Cosine Positional Encoding generator | None | `src/nn/pe.py` | Model Architecture |
| **07** | Multi-Head Attention | Solidify Queries, Keys, Values and Scaled Dot-Product logic | Implement pure PyTorch standard Multi-Head Attention | None | `src/nn/transformer.py` | Model Architecture |
| **08** | Transformer Encoder | Understand self-attention over flattened feature maps | Implement one robust Encoder block with feedforward | P06, P07 | `src/nn/transformer.py` | Model Architecture |
| **09** | Transformer Decoder | Understand object queries fetching visual properties | Implement a Decoder layer using Cross-Attention to image | P06, P07 | `src/models/dfine/decoder.py` | Model Architecture |
| **10** | HGNetV2-S Stem | Parse modern efficient CNN feature extractors | Build the Conv Stem for hierarchical patch extraction | None | `src/nn/backbone/hgnetv2.py` | Model Architecture |
| **11** | Model Assembly | Master the data flow traversing Core + Encoder + Decoder + Head | Link modules to form the complete `DFINE` forward pass | P08, P09, P10 | `src/models/dfine/dfine.py` | Capstone |
| **12** | Train & Eval Loop | Run the holistic learning pipeline | Implement the optimization loop with the `SetCriterion` | P05, P11 | `src/solver/`, `train.py` | Capstone |

## Phase 3: Paper-Compliant Innovations (Problems 13–15)
Advanced D-FINE innovations: matching consensus, distribution-based regression, multi-layer supervision.

| ID | Problem Name | Theory Goal | Coding Goal | Depends On | Repo Module Mapped | Innovation | Diagrams |
|---|---|---|---|---|---|---|---|
| **13** | Matching Union | Understand consensus matching (GO indices) across decoder layers | Implement matching deduplication and consensus voting | P03 | `src/models/dfine/matcher.py` | Multi-Layer Consensus | [Theory](problem_13/theory.md), [ARCH](ARCHITECTURE_DIAGRAMS.md#4-matching-union-go-indices-algorithm) |
| **14** | Fine-Grained Localization Loss | Master distribution-based bounding box regression | Compute soft labels via discrete distance bins and focal weighting | None | `src/models/dfine/loss.py` | D-FINE Core | [Theory](problem_14/theory.md), [ARCH](ARCHITECTURE_DIAGRAMS.md#5-fine-grained-localization-fgl-loss-flow) |
| **15** | Multi-Layer D-FINE Criterion | Assemble full paper-compliant D-FINE criterion | Integrate matcher, matching union, VFL loss, and FGL loss with multi-layer supervision | P05, P13, P14 | `src/models/dfine/criterion.py` | D-FINE Paper | [Theory](problem_15/theory.md), [ARCH](ARCHITECTURE_DIAGRAMS.md#3-multi-layer-supervision-flow) |

---

## 📊 Visual Resources

Complete architecture documentation with flowcharts, diagrams, and tensor flow visualizations:

### Text-Based Diagrams (ASCII Art)
- [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) - 9 comprehensive ASCII flowcharts
  - Overall D-FINE pipeline
  - Multi-layer decoder structure
  - Multi-layer supervision flow
  - Matching union algorithm
  - FGL loss computation flow
  - Loss computation per layer
  - Model output dictionary
  - Training loop flowchart
  - Tensor shape transformation

### Mermaid Flowcharts (Interactive)
- [MERMAID_FLOWCHARTS.md](MERMAID_FLOWCHARTS.md) - 12 interactive state diagrams
  - Overall training loop
  - DFINECriterion multi-layer flow
  - Matching union algorithm
  - Model forward (decoder layers)
  - Single layer matching process
  - VFL loss computation
  - FGL loss computation
  - Gradient flow through layers
  - Output dictionary structure
  - Multi-layer supervision philosophy
  - Data flow (batch processing)
  - Tensor shape transformation
