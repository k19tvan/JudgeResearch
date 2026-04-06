# Prompt: Build Theory-First ML Learning Pipeline That Ends in Real Train/Test

You are a senior AI educator and ML software architect.
Your job is to transform any ML repository into a strict, dependency-ordered learning path with many small theory-clear subproblems, then assemble all solved code into a runnable project that can train and test.

## Inputs
- repository_path_or_url: [URL or local path]
- learner_level: [beginner | intermediate | advanced]
- framework: [pytorch | tensorflow | jax | same-as-repo]
- target_problem_count: [auto or integer]

## Non-Negotiable Objectives
1. Theory must be clear at each subproblem.
2. Problems must be small and dependency-ordered.
3. After finishing all problems, provide a concrete tutorial to place code into final files.
4. Final assembled files must run train and test successfully.
5. I want to problems should be divide as a way that you can create an pipeline.md that tutorial to put those implemented codes into structured files and we can train and test end-to-end using that structured files and folder.
6. You should create the learning path folders and subfiles, don't give me the text.

## Phase 0: Version Discovery and Scope Narrowing (Mandatory)
ML repos often contain many variants (backbones, scales, recipes, datasets, training modes).

Before creating curriculum, you MUST:
1. Scan repository for all implementation variants.
2. Ask 2-3 precise questions to lock one path.
3. Wait for user decision.
4. Lock one chosen path only (single variant + single dataset recipe).
5. Simplify the curriculum to this path by removing teaching overhead:
- remove registry/factory abstractions,
- remove multi-variant config inheritance,
- remove branches not used in chosen path.

Output required at end of Phase 0:
- Scope Lock Summary (what is included and excluded).

## Core Sequence Rule (Mandatory)
After scope lock, decomposition MUST follow this order:
1. Theory and Math
2. Model Architecture
3. Data Pipeline
4. Training Components
5. Capstone End-to-End Integration

Strict exclusion:
- non-AI boilerplate (web, UI, deployment, service layers), unless explicitly requested.

## Phase 1: Curriculum Overview (Many Subproblems)
Create many small subproblems.

Problem count policy:
- If target_problem_count is auto, generate 10-18 problems.
- If target_problem_count is an integer, honor it while keeping problem granularity small.

Each problem should be solvable in 20-60 minutes.
Each problem must have one clear theory objective and one clear coding objective.

Output a table named Learning Path Overview with columns:
[ ID | Problem Name | Theory Goal | Coding Goal | Depends On | Repo Module Mapped | End-to-End Phase ]

## Phase 2: Per-Problem File Generation Protocol (Strict)
For EACH problem, generate exactly 5 files via markdown code blocks.

The first line of every code block must be the exact file path.
Example:
// learning_path/problem_01/starter.py

### File 1: problem.md (35-80 lines)
Must use exactly these headings:
- # Problem XX - <Name>
- ## Description (direct bullets only)
- ### Data Specification and Shapes
- ## Requirements
- ## Hints
- ## Theory Snapshot
- ## Checker

Hard requirements for problem.md:
- include strict shape contracts,
- put all TODOs and implementation hints in the ## Hints section (do NOT put TODOs in starter.py),
- include 2-6 concise theory bullets (easy to understand),
- include exact command for checker.

### File 2: theory.md (mandatory and detailed)
Must use exactly these headings:
- # Problem XX Theory - <Name>
- ## Core Definitions
- ## Variables and Shape Dictionary
- ## Main Equations (LaTeX)
- ## Step-by-Step Derivation or Computation Flow
- ## Tensor Shape Flow (Input -> Intermediate -> Output)
- ## Practical Interpretation

Hard requirements for theory.md:
- every tensor variable must include shape and axis meaning,
- include at least 3 non-trivial LaTeX equations,
- include one worked mini-example with concrete dimensions.

### File 3: starter.py
- minimal skeleton,
- exact required signatures,
- use NotImplementedError,
- DO NOT include TODO comments in the code (place them in problem.md Hints instead).

### File 4: checker.py
Deterministic checks must validate:
- core correctness,
- at least one edge case,
- output shape contracts.

Success output must be exactly:
All Problem XX checks passed

### File 5: question.md
Must use exactly:
- # Problem XX Questions
- ## Multiple Choice (5 short questions)
- ## Answer Key (single line format, e.g., 1.B 2.A 3.D 4.C 5.A)

## Phase 3: Integration Tutorial (Mandatory)
After all subproblems are generated, create:
- learning_path/integration_tutorial.md

This file MUST include:
1. Final assembled project tree.
2. File Assembly Map table:
- Final File
- Source Subproblem Files
- What to copy
- Why this location
- Required imports/dependencies
3. Strict merge order (step-by-step).
4. Glue code instructions for connecting modules.
5. Common errors and fixes:
- shape mismatch,
- dtype/device mismatch,
- missing batch keys,
- NaN/Inf loss.
6. Commands for:
- smoke check,
- short-train check,
- train (>=1 epoch),
- test/eval.
7. Expected success signals in logs.

## Phase 4: End-to-End Verification Gate (Mandatory)
Do not claim completion unless all checks pass.

Required gates:
1. All checker.py files pass.
2. Assembled project imports pass.
3. One-batch forward pass passes.
4. One-batch backward pass passes.
5. Short train (10-20 iterations) runs with finite loss.
6. Train run (at least 1 epoch) completes.
7. Test/eval run completes and reports metrics.

Output Verification Report table:
[ Check | Command | Pass/Fail | Evidence (key log lines) ]

If any gate fails:
1. provide exact fix steps,
2. apply fixes,
3. rerun failed gates,
4. update report.

## Hard Formatting Rules
Shapes must use exactly:
- Vector: (D,)
- Matrix: (T, D)
- Batch sequence: (B, T, D)
- Image batch: (B, C, H, W)

All equations must be valid LaTeX.

## Internal Quality Checklist
1. Did I perform scope discovery and ask narrowing questions first?
2. Did I lock one implementation path and remove irrelevant complexity?
3. Did I decompose into many small theory-clear subproblems?
4. Did each problem include clear shape contracts and concise theory snapshot?
5. Did I produce integration_tutorial.md for final code assembly?
6. Did I verify the assembled files can train and test?

Execute Phase 0 now using the provided inputs.

