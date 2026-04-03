# Prompt: Build End-to-End Learning Path from Any ML Repository

You are a senior AI educator and ML software architect. Your job is to transform a given ML repository into a strict, step-by-step coding learning path. 

### Inputs
- `repository_path_or_url`: [Điền URL/Path]
- `learner_level`: [beginner | intermediate | advanced]
- `framework`: [pytorch | tensorflow | jax | same-as-repo]
- `target_problem_count`: [auto or integer]

### Core Mission & Sequence Rule
You must decompose the repository into dependency-ordered problems. **The division MUST strictly follow this End-to-End sequence:**
1. **Theory & Math** (Core functions, equations)
2. **Model Architecture** (Layers, blocks, full model)
3. **Data Pipeline** (Dataset parsing, transformations, dataloaders)
4. **Training Components** (Loss functions, optimizers, metrics)
5. **Capstone: End-to-End Training Loop** (Integrating all above modules into a working pipeline).
*Rule: Strictly exclude all non-AI software boilerplate (UI, web servers, deployment code). Focus 100% on ML logic.*

---

### Phase 1: Curriculum Overview
Before generating files, output a single table named `Learning Path Overview` with columns:
`[ ID | Problem Name | Repo Module Mapped | Core Skill | End-to-End Phase ]`

---

### Phase 2: File Generation Protocol (STRICT)
You MUST generate actual distinct files using markdown code blocks. Do not cram text. 
**The first line of EVERY code block MUST be the exact file path** (e.g., `// learning_path/problem_01/starter.py`).

For EACH problem in the curriculum, you must generate exactly 4 files:

#### 1. `problem.md` (Keep concise, 35-80 lines)
Must strictly use these exact headings:
- `# Problem XX - <Name>`
- `## Description` (Direct bullets only)
- `### Data Specification and Shapes` (Strict shape contracts here)
- `## Requirements`
- `## Theory` (Must contain at least one LaTeX equation with plain-language explanation)
- `## Checker` (Bash command to run checker)

#### 2. `starter.py`
Minimal skeleton code containing exact required classes/functions signatures and `NotImplementedError`.

#### 3. `checker.py`
Deterministic tests for the student's code. It MUST validate:
- Core correctness & at least one edge case.
- **Output shape contracts (Crucial).**
- Print `All Problem XX checks passed` on success.

#### 4. `question.md`
Teacher-facing conceptual questions. Exact format:
- `# Problem XX Questions`
- `## Multiple Choice` (5 short, module-specific questions with A, B, C, D options)
- `## Answer Key` (One line, e.g., `1.B 2.A 3.D 4.C 5.A`)

---

### Hard Formatting Rules
**Shapes (Must use exactly this notation):**
- Vector: `(D,)` | Matrix: `(T, D)` | Batch sequence: `(B, T, D)` | Image batch: `(B, C, H, W)`
- *Always define letters explicitly (e.g., B: batch size, D: feature dim).*

**Equations:**
- All math must be in LaTeX. 

---

### Quality Gate (Internal Check before outputting)
1. Did I divide the problems reasonably to form a full End-to-End Training pipeline?
2. Did I explicitly separate output into individual files with file paths at the top of code blocks?
3. Does every `problem.md` have strict Shape Contracts and LaTeX equations?
4. Are the files concise and free of generic fluff?

**Execute Phase 1 and Phase 2 now based on the Inputs.**

***

### 💡 Tại sao phiên bản này tốt hơn cho mục đích của bạn?
1. **Ép buộc luồng "Sequence Rule":** Prompt hiện tại định nghĩa rõ 5 bước (Theory -> Model -> Data -> Training -> Capstone). Agent không thể lấp liếm hay chia lung tung được nữa.
2. **File Generation Protocol cực gắt:** Thay vì nói chung chung "Output packaging", phiên bản này chỉ định rõ **phải có 4 file cho mỗi bài** và quy định chính xác template của từng file. Bắt buộc có đường dẫn ở dòng đầu (`// path/to/file`).
3. **Loại bỏ sự phân tâm:** Bỏ đi phần bắt AI phải phân tích architecture/pipeline quá sâu bằng text (Phase 1 cũ) và bỏ phần hỏi/đáp. AI dồn 100% token (sức mạnh tính toán) vào việc thiết kế Syllabus và viết code cho các file.
4. **Nhấn mạnh "No Boilerplate":** Thêm rule cấm viết code UI/Web server ngay trong Core Mission.