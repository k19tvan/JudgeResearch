get_problems_from_repo_prompt = """
You are an expert Machine Learning Educator. Analyze the repository at '{repository_url}'.
The user wants to generate coding exercises to reconstruct this repository or learn its core components.

Target Student Level: {level}
Framework constraint: {framework}
User custom instructions: {user_note}

Based on the codebase structure, propose a progressive list of programming exercises (problems) that sequentially help a student build or understand this codebase.

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

feedback_prompt = """
  You are an ML Educator. Here is the current proposed exercise list for repository '{repo_url}':
  {current_problems}
  
  The user wants to modify this list with the following request:
  "{payload.feedback_text}"
  
  Please update the list of problems accordingly. Follow the exact same rules:
  - Output strictly a raw JSON list of objects.
  - No markdown formatting outside the JSON block.
  - Expected structure: [ {{"title": "...", "description": "...", "target_module": "..."}} ]
  """

# prompts/prompt.py

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

Please provide your output strictly in JSON format containing 5 keys: "statement", "theory", "tutorial", "solution", "coding".
Each value must be written in Markdown format (except "coding" and "solution", which should be standard Python code).

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
   - Detail the mathematical motivations, limitations of previous methods, and complete proofs using continuous math and LaTeX syntax (e.g., `$$\text{{GIoU}} = \text{{IoU}} - \frac{{\text{{Area}}(C) - U}}{{\text{{Area}}(C)}}$$`).

3. For "tutorial", you must follow the conceptual and vectorization/implementation guide structure:
   - Must use headings:
     - "#### **Conceptual Idea**"
     - Use bold 5th-level sub-headers: "##### **1. The Problem...**", "##### **2. The Solution...**"
     - Include clear ASCII-art structural diagrams illustrating bounding boxes, matrices, or tensor operations if applicable.
     - "##### **[Framework] Implementation ([Optimization Technique])**" (e.g., "##### **PyTorch Implementation (Broadcasting & Vectorization)**")
   - Explain the tensor mechanics step-by-step, detailing shapes (e.g., `Shape: (N, M, 2)`) and broadcasting logic, followed by highly-optimized vectorized code snippets.

4. For "solution", you must provide the complete, working solution code:
   - Must be executable Python code.
   - Should include comprehensive comments explaining the logic.
   - Must handle all edge cases and follow best practices.

5. For "coding", you must provide an executable-ready Python skeleton template:
   - Must import type-hinting components (`from typing import List, Tuple, Optional, Dict`).
   - Must include a rigorous docstring explaining all parameters, their shapes/dimensions, and return values, ending with a clean `pass` or `TODO` comment inside the function.

Expected JSON Structure:
{{
  "statement": "#### **Problem Description**\\nIn machine learning...",
  "theory": "#### **Theoretical Background: [Topic Title]**\\n...",
  "tutorial": "#### **Conceptual Idea**\\n...",
  "solution": "def solve():\\n    # Complete working solution\\n    pass",
  "coding": "from typing import List\\n\\ndef compute_metric():\\n    # TODO: Implement here\\n    pass"
}}
"""


correction_prompt = r"""
You are a high-precision JSON validation and syntax correction compiler. 
Your sole task is to take a malformed, broken, or syntactically invalid JSON string generated by another AI, and repair its formatting so that it is 100% valid JSON.

### TARGET SCHEMA CONSTRAINTS
The final repaired output must be a single, valid JSON object containing exactly these 4 keys:
1. "statement": Markdown string following the competitive-programming style.
2. "theory": Markdown string explaining ML/DL theory.
3. "tutorial": Markdown string for conceptual and implementation tutorials.
4. "coding": Clean, executable-ready Python code template.

### CRITICAL CORRECTION RULES (YOU MUST FOLLOW):
1. PRESERVE 100% CONTENT: Do NOT summarize, shorten, omit, or modify the actual words, educational explanations, mathematical proofs, LaTeX equations, or code logic. Only repair the JSON syntax.
2. ESCAPE INNER DOUBLE QUOTES: Convert any raw double quotes (") inside the text values (e.g., ...creating a "shortcut" for...) to escaped double quotes (\") or single quotes ('). Raw unescaped double quotes inside values are strictly illegal in JSON.
3. ESCAPE LATEX BACKSLASHES: Inside LaTeX formulas (e.g. \mu, \sigma, \text, \frac, \nabla, \epsilon), you must strictly escape all backslashes as double backslashes (\\mu, \\sigma, \\text, \\frac, \\nabla, \\epsilon) to avoid JSON invalid escape sequence errors.
4. ESCAPE RAW NEWLINES: Ensure all raw line breaks inside string values are properly escaped as "\n".
5. FIX STRUCTURE: Correct missing closing brackets/braces, fix trailing commas, and repair truncated JSON elements.

### OUTPUT FORMAT:
Output ONLY the valid JSON block wrapped in a standard ```json ... ``` markdown block. Do not add any introduction, explanations, or notes.

### MALFORMED JSON TO REPAIR:
{malformed_json}
"""