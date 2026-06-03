get_problems_from_repo_prompt = r"""
You are an expert Machine Learning Educator. Analyze the repository at '{repository_url}'.
The user wants to generate coding exercises to reconstruct this repository or learn its core components.

Target Student Level: {level}
Framework constraint: {framework}
User custom instructions: {user_note}

Based on the codebase structure, propose a progressive list of programming exercises (problems) that sequentially help a student build or understand this codebase.
Each exercise description must be mathematically and algorithmically rigorous, setting up a solid foundation for competitive-programming or advanced ML-course style detailed materials.

You MUST output your response strictly as a raw JSON list of objects. No markdown formatting outside the JSON block, no conversational text before or after.

Expected JSON Format:
[
  {{
    "title": "Exercise Title",
    "description": "A clear, detailed description of what the user needs to implement or complete.",
    "target_module": "Specific module or path in the repo (e.g., models/resnet.py)"
  }}
]
"""

feedback_prompt = r"""
You are an ML Educator. Here is the current proposed exercise list for repository '{repo_url}':
{current_problems}

The user wants to modify this list with the following request:
"{payload.feedback_text}"

Please update the list of problems accordingly. Follow the exact same rules:
- Output strictly a raw JSON list of objects.
- No markdown formatting outside the JSON block.
- Expected structure: [ {{"title": "...", "description": "...", "target_module": "..."}} ]
"""

create_detailedly_prompt = r"""
You are an expert Machine Learning Educator. Write complete, high-quality, step-by-step educational materials in Markdown format for the following exercise:
Title: "{title}"
Repository context: '{repository_url}'
Roadmap topic: "{roadmap_title}"
Target Test Cases Count: {num_test_cases}

Please provide your output strictly in JSON format containing exactly 7 keys: "statement", "theory", "tutorial", "solution", "coding", "checker", "test_inputs".
Each value must be written in Markdown format (except "coding", "solution", and "checker", which should be standard Python code, and "test_inputs" which must be a JSON array string).

You must enforce the strict formatting styles defined below for each key:

1. For "statement", you must follow the competitive-programming / rigorous ML-course structure:
   - Must use headings:
     - "#### **Problem Description**" (Introduction, mathematical background with LaTeX equations, formulations using $N$, $M$, etc.)
     - "#### **Input Specification**" (Format of input data, limits, constraints like $1 \le N \le 1000$ using LaTeX)
     - "#### **Output Specification**" (Format of output data, precision/error tolerances like $10^{{-6}}$ using LaTeX)
     - "#### **Example**" (With code blocks starting exactly with "**Input:**\n```text\n...\n```" and "**Output:**\n```text\n...\n```")
     - "#### **Note**" (Detailed mathematical tracing of the example case using LaTeX step-by-step equations)

2. For "theory", you must follow the formal theoretical background structure:
   - Must use headings:
     - "#### **Theoretical Background: [Topic Title]**"
     - Use bold 5th-level sub-headers: "##### **[Subtopic Title]**" (e.g., "##### **The Gradient Vanishing Limitation**", "##### **Properties of [Metric]:**")
   - Detail the mathematical motivations, limitations of previous methods, and complete proofs using continuous math and LaTeX syntax.

3. For "tutorial", you must follow the conceptual and vectorization/implementation guide structure:
   - Must use headings:
     - "#### **Conceptual Idea**"
     - Use bold 5th-level sub-headers: "##### **1. The Problem...**", "##### **2. The Solution...**"
     - Include clear ASCII-art structural diagrams illustrating bounding boxes, matrices, or tensor operations if applicable.
     - "##### **[Framework] Implementation ([Optimization Technique])**" (e.g., "##### **PyTorch Implementation (Broadcasting & Vectorization)**")
   - Explain the tensor mechanics step-by-step, detailing shapes (e.g., `Shape: (N, M, 2)`) and broadcasting logic, followed by highly-optimized vectorized code snippets.

4. For "solution", you must provide the complete, working solution code:
   - Must be executable Python code.
   - It must read input from "input.json" and write the computed results back to "output.json".
   - Should include comprehensive comments explaining the logic.

5. For "coding", you must provide an executable-ready Python skeleton template for the student:
   - Must import type-hinting components (`from typing import List, Tuple, Optional, Dict`).
   - Must include a rigorous docstring explaining all parameters, their shapes/dimensions, and return values, ending with a clean `pass` or `TODO` comment inside the function.
   - Must include the identical JSON IO boilerplate reading from "input.json" and writing to "output.json" as in the "solution" key, leaving only the target core function as `pass`.

6. For "checker", you must provide a python validation script:
   - Contains assertion blocks verifying that the candidate's output is numerically correct and matches expected bounds.

7. For "test_inputs", you must provide exactly {num_test_cases} testcases:
   - Must be structured as a serialized JSON array string containing exactly {num_test_cases} distinct objects.
   - Each object represents the complete raw dictionary to be dumped into "input.json".

---

### FEW-SHOT EXAMPLE FOR YOUR REFERENCE:
Here is how you should structure the output JSON for a problem like "Generalized Intersection over Union (GIoU) Matrix":

{{
  "statement": "#### **Problem Description**\\nIn object detection, evaluating localization accuracy is crucial. You are tasked with implementing the **Generalized Intersection over Union (GIoU)** matrix computation between $N$ predicted bounding boxes and $M$ ground-truth bounding boxes.\\n\\n#### **Input Specification**\\nInput is read from `input.json` containing:\\n- `pred_boxes`: List of $N$ lists, each representing a box $[x_{{min}}, y_{{min}}, x_{{max}}, y_{{max}}]$\\n- `gt_boxes`: List of $M$ lists, each representing a box $[x_{{min}}, y_{{min}}, x_{{max}}, y_{{max}}]$\\nConstraints: $1 \\le N, M \\le 1000$.\\n\\n#### **Output Specification**\\nOutput must be written to `output.json` as a dictionary containing:\\n- `giou_matrix`: An $N \\times M$ matrix where the element at $(i, j)$ is the GIoU score bounded in $[-1, 1]$. Precision tolerance: $10^{{-6}}$.",
  "theory": "#### **Theoretical Background: IoU vs. GIoU**\\nIn classical Object Detection evaluation, **Intersection over Union (IoU)** is defined as:\\n$$\\text{{IoU}}(A, B) = \\frac{{|A \\cap B|}}{{|A \\cup B|}}$$\\n\\n##### **The Gradient Vanishing Limitation**\\nWhen predicted box $A$ and ground truth box $B$ do not overlap, $\\text{{IoU}}(A, B) = 0$. Consequently, the gradient of the IoU-based loss function is:\\n$$\\nabla \\mathcal{{L}}_{{IoU}} = 0$$\\n\\n##### **The GIoU Solution**\\nGeneralized Intersection over Union (GIoU) introduces a penalty term using the smallest enclosing box $C$:\\n$$\\text{{GIoU}}(A, B) = \\text{{IoU}}(A, B) - \\frac{{\\text{{Area}}(C \\setminus (A \\cup B))}}{{\\text{{Area}}(C)}}$$",
  "tutorial": "#### **Conceptual Idea**\\n##### **1. The Problem**\\nEvaluating $N \\times M$ combinations sequentially using Python loops creates significant computational overhead.\\n\\n##### **2. The Solution (PyTorch Broadcasting)**\\nWe can expand the tensors to leverage PyTorch's optimized backend:\\n* Expand `pred_boxes` to shape $(N, 1, 4)$\\n* Expand `gt_boxes` to shape $(1, M, 4)$\\n\\n```text\\n+-------------------+ (Smallest Enclosing Box C)\\n|  +-------+        |\\n|  | Box A |        |\\n|  +-------+        |\\n|                   |\\n|          +-----+  |\\n|          | Box B|  |\\n|          +-----+  |\\n+-------------------+\\n```",
  "solution": "import json\\nimport sys\\nimport torch\\n\\ndef compute_giou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:\\n    pred = pred_boxes[:, None, :]\\n    gt = gt_boxes[None, :, :]\\n\\n    inter_xmin = torch.maximum(pred[..., 0], gt[..., 0])\\n    inter_ymin = torch.maximum(pred[..., 1], gt[..., 1])\\n    inter_xmax = torch.minimum(pred[..., 2], gt[..., 2])\\n    inter_ymax = torch.minimum(pred[..., 3], gt[..., 3])\\n\\n    inter_w = (inter_xmax - inter_xmin).clamp(min=0)\\n    inter_h = (inter_ymax - inter_ymin).clamp(min=0)\\n    inter_area = inter_w * inter_h\\n\\n    pred_area = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (pred[..., 3] - pred[..., 1]).clamp(min=0)\\n    gt_area = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (gt[..., 3] - gt[..., 1]).clamp(min=0)\\n\\n    union_area = pred_area + gt_area - inter_area\\n    eps = 1e-7\\n    iou = inter_area / (union_area + eps)\\n\\n    enc_xmin = torch.minimum(pred[..., 0], gt[..., 0])\\n    enc_ymin = torch.minimum(pred[..., 1], gt[..., 1])\\n    enc_xmax = torch.maximum(pred[..., 2], gt[..., 2])\\n    enc_ymax = torch.maximum(pred[..., 3], gt[..., 3])\\n\\n    enc_w = (enc_xmax - enc_xmin).clamp(min=0)\\n    enc_h = (enc_ymax - enc_ymin).clamp(min=0)\\n    enc_area = enc_w * enc_h\\n\\n    giou = iou - (enc_area - union_area) / (enc_area + eps)\\n    return giou\\n\\nif __name__ == \\\"__main__\\\":\\n    with open(\\\"input.json\\\", \\\"r\\\") as f:\\n        input_data = json.load(f)\\n    \\n    pred_boxes = torch.tensor(input_data.get(\\\"pred_boxes\\\", []), dtype=torch.float32)\\n    gt_boxes = torch.tensor(input_data.get(\\\"gt_boxes\\\", []), dtype=torch.float32)\\n\\n    result_tensor = compute_giou_matrix(pred_boxes, gt_boxes)\\n    result = result_tensor.tolist() if isinstance(result_tensor, torch.Tensor) else []\\n        \\n    with open(\\\"output.json\\\", \\\"w\\\") as f:\\n        json.dump({{\\\"giou_matrix\\\": result}}, f)",
  "coding": "import json\\nimport sys\\nimport torch\\n\\ndef compute_giou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:\\n    \\\"\\\"\\\"\\n    Computes the GIoU matrix between N predicted boxes and M ground truth boxes.\\n    \\\"\\\"\\\"\\n    pass\\n\\nif __name__ == \\\"__main__\\\":\\n    with open(\\\"input.json\\\", \\\"r\\\") as f:\\n        input_data = json.load(f)\\n    \\n    pred_boxes = torch.tensor(input_data.get(\\\"pred_boxes\\\", []), dtype=torch.float32)\\n    gt_boxes = torch.tensor(input_data.get(\\\"gt_boxes\\\", []), dtype=torch.float32)\\n\\n    result_tensor = compute_giou_matrix(pred_boxes, gt_boxes)\\n    result = result_tensor.tolist() if isinstance(result_tensor, torch.Tensor) else []\\n        \\n    with open(\\\"output.json\\\", \\\"w\\\") as f:\\n        json.dump({{\\\"giou_matrix\\\": result}}, f)",
  "checker": "import torch\\n\\ndef check_output(candidate_matrix, expected_matrix, tolerance=1e-5):\\n    assert candidate_matrix.shape == expected_matrix.shape, f\\\"Shape mismatch\\\"\\n    diff = torch.abs(candidate_matrix - expected_matrix)\\n    assert torch.all(diff <= tolerance), f\\\"Numerical discrepancy exceeds limit\\\"\\n    print(\\\"Passed\\\")",
  "test_inputs": "[{{\\\"pred_boxes\\\": [[0.0, 0.0, 2.0, 2.0]], \\\"gt_boxes\\\": [[1.0, 1.0, 3.0, 3.0]]}}, {{\\\"pred_boxes\\\": [[0.0, 0.0, 1.0, 1.0]], \\\"gt_boxes\\\": [[2.0, 2.0, 3.0, 3.0]]}}]"
}}

---

Now, based on the target configuration below, generate the corresponding high-quality educational materials:
Title: "{title}"
Repository: "{repository_url}"
Topic: "{roadmap_title}"
Number of test cases requested: {num_test_cases}

Do not add any conversational text or explanation outside the valid JSON block.
"""

validate_problem_from_repo_prompt = r"""

You are a strict JSON Validator and Senior Deep Learning Engineer. Your task is to ingest a raw, potentially malformed or technically inaccurate JSON string representing a roadmap of programming tasks, repair it, and output a standardized, production-ready JSON array.

### Your Objectives:

1. **Syntax & Formatting Repair**:
   - Detect and fix any syntax errors (e.g., missing commas, unescaped double quotes inside descriptions, trailing commas, mismatched brackets, or unescaped newline characters).
   - Ensure the output is a perfectly valid JSON array of objects.

2. **Technical Correction**:
   - Check the technical accuracy of the deep learning concepts, PyTorch references, and module structures inside each object.
   - If there is an inaccurate technical term, mathematical error, or confusing description, refine it to be scientifically accurate while keeping the original context and intention.

3. **Schema Enforcement**:
   - Each object in the array MUST contain exactly these three keys:
     - "title": (String) Concise, professional technical title.
     - "description": (String) Clearly defined task requirements, parameter lists, and expected behavior.
     - "target_module": (String) Reference to the PyTorch module, class, or script path.
   - Do not allow any extra keys or nested structures.

4. **Response Format Constraint**:
   - Output ONLY the clean, verified, and parsed JSON array.
   - Do not write any conversational introduction, notes, markdown explanation, or post-text. The response must start with `[` and end with `]`.

---
### RAW INPUT JSON TO REPAIR AND NORMALIZE:

{last_ai_response}

"""