import os
import json

base_dir = "/home/enn/workspace/project/AI_Judge/example_researches/D-FINE/learning_path"

problems = [
    {"id": "04", "title": "Hybrid Encoder Projectors", "module": "hybrid_encoder.py", "desc": "Mapping multi-scale CNN backbones to uniform hidden dimensions."},
    {"id": "05", "title": "BiFPN Feature Fusion", "module": "hybrid_encoder.py", "desc": "Top-Down and Bottom-Up cross-scale interaction using CSP PANet."},
    {"id": "06", "title": "Contrastive Denoising (CDN) Concept", "module": "denoising.py", "desc": "Adding noise to true boxes to accelerate DETR convergence via auxiliary tasks."},
    {"id": "07", "title": "D-FINE Positional Encoding", "module": "utils.py", "desc": "Converting spatial coordinates into continuous query sine embeddings."},
    {"id": "08", "title": "D-FINE Decoder Cross-Attention", "module": "dfine_decoder.py", "desc": "Simplified multi-head cross-attention layer between queries and multi-scale features."},
    {"id": "09", "title": "Fine-Grained Box Refinement", "module": "dfine_decoder.py", "desc": "The core D-FINE novelty: distribution-based regressor mapping offset distributions to coordinates."},
    {"id": "10", "title": "Decoupled Decoder Heads", "module": "dfine_decoder.py", "desc": "Separating features for Classification vs. Regression metrics in the final output heads."},
    {"id": "11", "title": "Bipartite Matching Cost Matrix", "module": "matcher.py", "desc": "Pair-wise Class, Bbox, and GIoU cost matrices for Hungarian assignment."},
    {"id": "12", "title": "Hungarian Matching Indices", "module": "matcher.py", "desc": "Solving the global optimum for set assignment natively with scipy."},
    {"id": "13", "title": "Set Criterion (Main Loss)", "module": "dfine_criterion.py", "desc": "Computing aggregate loss using target permutation indices matching targets against predictions."},
    {"id": "14", "title": "Set Criterion (Denoising Aux Loss)", "module": "dfine_criterion.py", "desc": "Integrating CDN contrastive reconstruction losses on top of the main Hungarian loss."},
    {"id": "15", "title": "Post-processing & Top-K Extraction", "module": "postprocessor.py", "desc": "Mapping raw logits and scaled coordinates into final bounding box predictions with confidence thresholds."},
    {"id": "16", "title": "End-to-End D-FINE Forward Pass", "module": "dfine.py", "desc": "The complete graph assembly connecting Encoder, Denoise, Decoder, and Loss."}
]

for p in problems:
    folder = os.path.join(base_dir, f"problem_{p['id']}")
    os.makedirs(folder, exist_ok=True)
    
    # problem.md
    with open(os.path.join(folder, "problem.md"), "w") as f:
        f.write(f"""# Problem {p['id']} - {p['title']}

## Description
- {p['desc']}
- This module corresponds to `{p['module']}` in the final D-FINE repository phase.
- Your task is to implement the core mechanics according to the shape constraints.

### Data Specification and Shapes
- Read the theory file, this varies by implementation.

## Requirements
- Follow the signature provided in `starter.py`.

## Hints
- Replace `NotImplementedError` with the mathematical tensor computations.

## Theory Snapshot
- Ensure inputs and targets match exactly according to the constraints in the paper.

## Checker
Run the provided checker to validate your implementation:
`python checker.py`
""")

    # theory.md
    with open(os.path.join(folder, "theory.md"), "w") as f:
        f.write(f"""# Problem {p['id']} Theory - {p['title']}

## Core Definitions
{p['desc']}

## Variables and Shape Dictionary
| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| `input` | `(B, C, H, W)` | Standard feature map block. |

## Main Equations (LaTeX)
$$ Y = \\text{{Layer}}(X) $$
*(Specific equations will be populated during the detailed problem phase).*

## Step-by-Step Derivation or Computation Flow
1. Load features.
2. Apply transformation.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- `Input`: Tensor `(B, *dims)`
- `Output`: Transformed Tensor `(B, *new_dims)`

## Practical Interpretation
This module handles a crucial step in the end-to-end target sequence.
""")

    # starter.py
    with open(os.path.join(folder, "starter.py"), "w") as f:
        f.write(f"""import torch
import torch.nn as nn
import torch.nn.functional as F

class Module{p['id']}(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError("Implement {p['title']}")
""")

    # checker.py
    with open(os.path.join(folder, "checker.py"), "w") as f:
        f.write(f"""import torch
from starter import Module{p['id']}

def run_checks():
    model = Module{p['id']}()
    print("All Problem {p['id']} checks passed")

if __name__ == "__main__":
    run_checks()
""")

    # question.md
    with open(os.path.join(folder, "question.md"), "w") as f:
        f.write(f"""# Problem {p['id']} Questions

## Multiple Choice
1. What is the fundamental purpose of this subproblem?
A. Data augmentation.
B. To handle {p['title']} logic.
C. Deploying the model.
D. Extracting logs.

## Answer Key
1.B
""")

print("Successfully generated all problem templates.")
