import datetime
import secrets
import sqlite3
import jwt
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.staticfiles import StaticFiles
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ValidationError
from backend.auth import (
    ALGORITHM,
    SECRET_KEY,
    hash_password,
    verify_password,
    create_access_token,
    validate_password,
    validate_username,
)
from typing import List, Optional
from backend.file_manager import initialize_problem_storage, save_and_unzip_file, validate_folder_structure
from time import perf_counter
from backend.sandbox import execute_user_code, compare_outputs, extract_testcases_from_folders
import time
import traceback
import zipfile
import shutil
import re
import os
import requests  
import json
from prompts.prompt import get_problems_from_repo_prompt, feedback_prompt, create_detailedly_prompt
from dotenv import load_dotenv
from json_repair import repair_json
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends
from typing import Optional
import os
import sqlite3
import os
import sys
import time
import json
import sqlite3
import tempfile
import subprocess
from typing import Optional, Any
from typing import List, Optional
from backend.file_manager import initialize_problem_storage, save_and_unzip_file, validate_folder_structure
from time import perf_counter
from backend.sandbox import execute_user_code, compare_outputs, extract_testcases_from_folders
import time
import traceback
import zipfile
import shutil
import re
import os
import requests  
import json
from prompts.prompt import get_problems_from_repo_prompt, feedback_prompt, create_detailedly_prompt
from dotenv import load_dotenv
from json_repair import repair_json
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends
from typing import Optional
import os
import sqlite3
import os
import sys
import time
import json
import sqlite3
import tempfile
import subprocess
from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel
from typing import Any, List, Dict
from fastapi import Body


os.makedirs("database", exist_ok=True)
os.makedirs("storage/problems", exist_ok=True)
os.makedirs("storage/avatars", exist_ok=True)

app = FastAPI()
app.mount("/avatars", StaticFiles(directory="storage/avatars"), name="avatars")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:21080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_path = "storage/"
deepwiki_url = "http://localhost:21082"
load_dotenv()
EMAIL_FORMAT_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def ensure_schema_migrations():
    conn = sqlite3.connect("database/database.db", check_same_thread=False)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(draft_problem_sessions)")
        draft_columns = {row[1] for row in cursor.fetchall()}
        if draft_columns and "research_title" not in draft_columns:
            cursor.execute("ALTER TABLE draft_problem_sessions ADD COLUMN research_title TEXT")
        conn.commit()
    finally:
        conn.close()

ensure_schema_migrations()

# ================= DATABASE CONNECTION DEPENDENCY =================
def get_db():
    """
    FastAPI Dependency: Tạo kết nối DB mới cho mỗi request và tự động đóng 
    khi request hoàn tất xử lý.
    """
    conn = sqlite3.connect("database/database.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close() # Đóng kết nối an toàn cho riêng request này

# ================= PYDANTIC SCHEMAS =================
class UserRegister(BaseModel):
    username: str
    password: str
    display_name: str
    email: EmailStr
    
class UserLogin(BaseModel):
    username: str
    password: str

class LogoutRequest(BaseModel):
    refresh_token: str

class ManualProblemCreate(BaseModel):
    name: str
    source: Optional[str] = None
    statement_markdown: str     
    theory_markdown: Optional[str] = "" 
    tutorial_markdown: Optional[str] = ""  
    solution_markdown: Optional[str] = ""
    coding_markdown: Optional[str] = ""
    checker_markdown: Optional[str] = ""
    author_id: int            
    
class FeedbackRequest(BaseModel):
    session_id: int
    feedback_text: str
    
class FinalizeSessionRequest(BaseModel):
    session_id: int
    roadmap_title: str

class ApproveProblemRequest(BaseModel):
    approve: bool  # True: duyệt cho công khai, False: từ chối duyệt (trả về nháp)
    
class DetailedProblemMaterials(BaseModel):
    statement: str = Field(..., description="Markdown description of the problem and requirements")
    theory: str = Field(..., description="Markdown DL/ML theory basis")
    tutorial: str = Field(..., description="Markdown step-by-step guidance on how to solve the problem")
    solution: str = Field(..., description="Complete solution code")
    coding: str = Field(..., description="Python template or code file content")
    checker: str = Field(..., description="Checker code for validating submissions")
    # Thêm trường này để nhận 3 bộ input mẫu dạng chuỗi JSON từ AI
    test_inputs: str = Field(..., description="A JSON array string containing 3 testcase input objects matching the input.json schema")

class SubmissionCreate(BaseModel):
    problem_id: int
    submitted_code: str

class UserProfileUpdate(BaseModel):
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    avatar_url: Optional[str] = None

class AccountDeactivationRequest(BaseModel):
    confirm: bool

class RunRequest(BaseModel):
    submitted_code: str

class SubmitRequest(BaseModel):
    user_id: int
    submitted_code: str

class MakeAdminRequest(BaseModel):
    user_id: int
    secret_key: str

class RequestApprovalPayload(BaseModel):
    user_id: int

class AdminActionPayload(BaseModel):
    admin_id: int

class ProblemsFromRepo(BaseModel):
    roadmap_name: str
    repository_url: str
    level: str
    user_id: int
    user_note: Optional[str] = ""
    framework: Optional[str] = ""
    num_test_cases: Optional[int] = 3 # Thêm trường này để nhận từ Form gửi lên
    
class ProposedProblem(BaseModel):
    title: str = Field(..., description="Problem title, a concise name for the exercise")
    description: str = Field(..., description="Detailed description of the problem, outlining what the user needs to implement or understand")
    target_module: str = Field(..., description="Path to the target source module in the repository")

class ProblemsFromRepoDataResponse(BaseModel):
    session_id: int
    repository_url: str
    roadmap_name: str
    proposed_problems: List[ProposedProblem]

class ProblemsFromRepoResponse(BaseModel):
    status: str
    message: str
    data: ProblemsFromRepoDataResponse

def check_wiki_cache(owner: str, repo: str, repo_type: str = "github", language: str = "en"):  
    """Check if a repository is cached."""  
    params = {  
        "owner": owner,  
        "repo": repo,  
        "repo_type": repo_type,  
        "language": language  
    }  
    response = requests.get(f"{deepwiki_url}/api/wiki_cache", params=params)  
    return response.json() if response.status_code == 200 else None  

def ask_question(repo_url: str, question: str, provider: str = "google"):  
    """Ask a question about a repository using the streaming endpoint."""  
    payload = {  
        "repo_url": repo_url,  
        "messages": [{"role": "user", "content": question}],  
        "provider": provider  
    }  
      
    response = requests.post(f"{deepwiki_url}/chat/completions/stream", json=payload, stream=True)  
      
    # Process streaming response  
    full_response = ""  
    for chunk in response.iter_content(chunk_size=None):  
        if chunk:  
            full_response += chunk.decode('utf-8')  
      
    return full_response

def get_repo_owner_and_name(repo_url: str):
    pattern = r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)"
    match = re.search(pattern, repo_url)
    if match:
        return match.group("owner"), match.group("repo")
    else:
        raise ValueError("Invalid GitHub repository URL")

def validate_and_ensure_complete_materials(data: dict) -> dict:
    """
    Validate that parsed JSON has all required fields: statement, theory, tutorial, solution, coding.
    If any field is missing or empty, log error details for debugging.
    """
    required_fields = {"statement", "theory", "tutorial", "solution", "coding"}
    missing_fields = required_fields - set(data.keys())
    
    if missing_fields:
        present_fields = set(data.keys())
        print(f"\n[ERROR] Missing required fields: {missing_fields}")
        print(f"[ERROR] Present fields: {present_fields}")
        print(f"[DEBUG] Parsed JSON structure (first 200 chars of each field):")
        for key in present_fields:
            value_preview = str(data.get(key, ""))[:200]
            print(f"  - {key}: {value_preview}...")
        
        raise ValueError(
            f"AI response JSON is incomplete. Missing fields: {', '.join(sorted(missing_fields))}. "
            f"Got only: {', '.join(sorted(present_fields))}. "
            f"This usually means the AI response was truncated or malformed. "
            f"Try again or check the raw response in the debug logs."
        )
    
    # Validate each field has content
    empty_fields = {k: v for k, v in data.items() if not v or (isinstance(v, str) and len(v.strip()) == 0)}
    if empty_fields:
        print(f"\n[ERROR] Empty required fields: {list(empty_fields.keys())}")
        raise ValueError(
            f"AI response has empty required fields: {', '.join(empty_fields.keys())}. "
            f"Please try generating materials again."
        )
    
    return data


def parse_and_repair_json(raw_text: str) -> dict:
    """
    Hệ thống tự vá lỗi JSON (Self-Healing Parser):
    1. Làm sạch thô bằng Regex.
    2. Thử parse thông thường.
    3. Thử vá lỗi bằng thư viện json-repair (miễn phí, tốc độ cao).
    
    ALWAYS returns a dict, never a list.
    If JSON is an array, tries to extract first element if it's a dict.
    """
    
        
    pattern = r'"(statement|theory|tutorial|solution|coding|checker|test_inputs)"\s*:\s*"((?:[^"\\]|\\.)*)"'
    matches = re.findall(pattern, raw_text)
    extracted_fields = {}
    for key, value in matches:
        decoded_value = value.encode('utf-8').decode('unicode_escape')
        extracted_fields[key] = decoded_value

    if "test_inputs" in extracted_fields and not isinstance(extracted_fields["test_inputs"], str):
        extracted_fields["test_inputs"] = json.dumps(extracted_fields["test_inputs"])

    return extracted_fields


def clean_json_response(raw_text: str) -> str:
    """
    Clean AI/LLM textual responses to extract the JSON payload.
    Handles fenced code blocks (```json ... ```), generic code fences, or
    extracts the first JSON object/array found in the text.
    Returns a JSON string ready for json.loads() or raises ValueError when nothing found.
    """
    if not raw_text:
        raise ValueError("Empty AI response")

    text = raw_text.strip()

    # Prefer explicit ```json fenced blocks
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()

    # Fallback: any fenced code block
    m = re.search(r"```(?:[^\n]*)\n(.*?)\n```", text, flags=re.S)
    if m:
        return m.group(1).strip()

    # Fallback: extract first {...} or [...] substring
    m = re.search(r"(\{(?:.|\n)*?\}|\[(?:.|\n)*?\])", text, flags=re.S)
    if m:
        return m.group(1).strip()

    # Last resort: attempt to repair/clean simple line-delimited JSON using json_repair
    try:
        repaired = repair_json(text)
        return repaired
    except Exception:
        raise ValueError("Unable to extract JSON payload from AI response")

def compare_approx(v1: Any, v2: Any, tolerance: float = 1e-6) -> bool:
    # 1. Nếu cả hai là Danh sách (List) -> Duyệt đệ quy từng phần tử
    if isinstance(v1, list) and isinstance(v2, list):
        if len(v1) != len(v2):
            return False
        return all(compare_approx(x, y, tolerance) for x, y in zip(v1, v2))
    
    # 2. Nếu cả hai là Đối tượng (Dict/JSON) -> So sánh các trường (Keys) và duyệt đệ quy từng trường
    elif isinstance(v1, dict) and isinstance(v2, dict):
        # Đảm bảo cả hai có cùng tập hợp các trường (Keys)
        if set(v1.keys()) != set(v2.keys()):
            return False
        # So sánh đệ quy giá trị của từng trường
        return all(compare_approx(v1[k], v2[k], tolerance) for k in v1)
    
    # 3. Nếu cả hai là Số (Float, Int) -> So sánh có sai số
    elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return abs(v1 - v2) / max(1.0, abs(v2)) <= tolerance
    
    # 4. Các kiểu dữ liệu khác (String, Boolean, None) -> So sánh bằng trực tiếp
    return v1 == v2

def extract_python_code(markdown_content: Optional[str]) -> str:
    """
    Trích xuất mã nguồn Python sạch từ khối Markdown code block của AI.
    """
    if not markdown_content:
        return ""
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", markdown_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return markdown_content.strip()

def validate_email_format(email: str) -> Optional[str]:
    if not EMAIL_FORMAT_PATTERN.match(email):
        return "Email must be a valid format, e.g., example@domain.com."
    return None

# ================= API ENDPOINTS =================

@app.post("/api/auth/register")
def register(user: UserRegister, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()

    username = user.username.strip()
    display_name = user.display_name.strip()
    email = user.email.strip()
    password = user.password

    missing_fields = []
    if not username:
        missing_fields.append("username")
    if not display_name:
        missing_fields.append("display_name")
    if not email:
        missing_fields.append("email")
    if not password or not password.strip():
        missing_fields.append("password")

    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise HTTPException(status_code=400, detail=f"Missing required fields: {missing_text}")

    password_errors = validate_password(password)
    if password_errors:
        raise HTTPException(status_code=400, detail=" ".join(password_errors))

    cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="Username already exists")

    cursor.execute("SELECT 1 FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="Email already exists")

    hashed_password = hash_password(password)
    
    try:
        cursor.execute(
            """INSERT INTO users (username, password_hash, display_name, email) VALUES (?, ?, ?, ?)""", 
            (username, hashed_password, display_name, email)
        )
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")

    return {"message": "Registration successful"}
        
@app.post("/api/auth/login")
def login(credentials: UserLogin, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (credentials.username,))
    user = cursor.fetchone()
    
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"], "user_id": user["id"]},
        expires_delta=datetime.timedelta(minutes=30)
    )
    
    refresh_token = secrets.token_hex(32)
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
    
    cursor.execute(
        "INSERT INTO refresh_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user["id"], refresh_token, expires_at.strftime('%Y-%m-%d %H:%M:%S'))
    )
    db.commit()
    
    # CẬP NHẬT: Trả thêm "user_role" và "user_id" về cho Frontend
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user_id": user["id"],
        "user_role": user["role"]  # Thêm dòng này để truyền vai trò ('admin'/'user')
    }
    
    
@app.post("/api/auth/refresh")
def refresh(payload: LogoutRequest, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    
    cursor.execute(
        "SELECT * FROM refresh_tokens WHERE token = ? AND expires_at > DATETIME('now')", 
        (payload.refresh_token,)
    )
    token_record = cursor.fetchone()
    if not token_record:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    
    cursor.execute("SELECT id, username, role FROM users WHERE id = ?", (token_record["user_id"],))
    user = cursor.fetchone()
    
    new_access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"], "user_id": user["id"]},
        expires_delta=datetime.timedelta(minutes=30)
    )
    
    return {"access_token": new_access_token, "token_type": "bearer"}

@app.post("/api/auth/logout")
def logout(payload: LogoutRequest, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("DELETE FROM refresh_tokens WHERE token = ?", (payload.refresh_token,))
    db.commit()
    return {"message": "Logged out successfully"}

def normalize_problem_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    return re.sub(r'[-\s]+', '_', name).strip('_')


# Usecase : Create problem manually
@app.post("/api/problems/create/manual")
async def create_problem_manual(
    name: str = Form(...),
    source: Optional[str] = Form(None),
    statement_markdown: str = Form(...),
    theory_markdown: Optional[str] = Form(""),
    tutorial_markdown: Optional[str] = Form(""),
    solution_markdown: Optional[str] = Form(""),
    coding_markdown: Optional[str] = Form(""),
    checker_markdown: Optional[str] = Form(""),
    author_id: int = Form(...),
    input_zip: Optional[UploadFile] = File(None),   
    output_zip: Optional[UploadFile] = File(None), 
    db: sqlite3.Connection = Depends(get_db)
):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Problem name cannot be empty")
    if not statement_markdown.strip():
        raise HTTPException(status_code=400, detail="Problem statement cannot be empty")

    cursor = db.cursor()
    try:
        name_slug = normalize_problem_name(name)
        cursor.execute("SELECT id FROM problems WHERE name = ?", (name_slug,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="The problem name already exists. Please choose a different name.")

        problem_folder = initialize_problem_storage(storage_path, name_slug)

        statement_path = os.path.join(problem_folder, "statement.md")
        theory_path = os.path.join(problem_folder, "theory.md") if theory_markdown else None
        tutorial_path = os.path.join(problem_folder, "tutorial.md") if tutorial_markdown else None
        solution_path = os.path.join(problem_folder, "solution.py") if solution_markdown else None
        coding_path = os.path.join(problem_folder, "coding.py") if coding_markdown else None
        checker_path = os.path.join(problem_folder, "checker.py") if checker_markdown else None

        with open(statement_path, "w", encoding="utf-8") as f:
            f.write(statement_markdown)

        if theory_path:
            with open(theory_path, "w", encoding="utf-8") as f:
                f.write(theory_markdown)

        if tutorial_path:
            with open(tutorial_path, "w", encoding="utf-8") as f:
                f.write(tutorial_markdown)

        if solution_path:
            with open(solution_path, "w", encoding="utf-8") as f:
                f.write(solution_markdown)

        if coding_path:
            with open(coding_path, "w", encoding="utf-8") as f:
                f.write(coding_markdown)

        if checker_path:
            with open(checker_path, "w", encoding="utf-8") as f:
                f.write(checker_markdown)

        input_folder_path = None
        output_folder_path = None
        
        # Sử dụng đúng tên biến tệp tin nhận từ tham số hàm
        if input_zip:
            input_folder_path = save_and_unzip_file(problem_folder, input_zip, "inputs")
            if not validate_folder_structure(input_folder_path):
                raise HTTPException(status_code=400, detail="Invalid input folder structure or missing .txt files")
                
        if output_zip:
            output_folder_path = save_and_unzip_file(problem_folder, output_zip, "outputs")
            if not validate_folder_structure(output_folder_path):
                raise HTTPException(status_code=400, detail="Invalid output folder structure or missing .txt files")

        # Thêm checker_path vào câu lệnh INSERT để lưu vào cơ sở dữ liệu
        cursor.execute("""
            INSERT INTO problems (
                name, source, statement_path, theory_path, tutorial_path, 
                solution_path, coding_path, checker_path, author_id, 
                is_public, request_status, input_folder_path, output_folder_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'NONE', ?, ?)
        """, (
            name,
            source,
            statement_path,
            theory_path,
            tutorial_path,
            solution_path,
            coding_path,
            checker_path,
            author_id,
            input_folder_path,
            output_folder_path
        ))

        db.commit()
        problem_id = cursor.lastrowid

        return {
            "status": "success",
            "message": "Create problem manually successfully",
            "data": {
                "id": problem_id,
                "name": name,
                "statement_path": statement_path,
                "theory_path": theory_path,
                "tutorial_path": tutorial_path,
                "solution_path": solution_path,
                "coding_path": coding_path,
                "checker_path": checker_path, 
                "author_id": author_id,
                "is_public": 0,
                "request_status": "NONE",
                "input_folder_path": input_folder_path, 
                "output_folder_path": output_folder_path 
            }
        }

    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
    except IOError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"File write error: {str(e)}")

# Usecase: Fetch and filter problems
@app.get("/api/problems/filter")
def filter_problems(user_id: int = None, filter_mode: str = "public", db: sqlite3.Connection = Depends(get_db)):
    """
    Filter problems by mode with user-specific submission status:
    - public: Only public problems (is_public = 1)
    - private: Only user's private problems (is_public = 0, author_id = user_id)
    - all: All problems (requires user_id)
    """
    cursor = db.cursor()
    try:
        # Ở đây chúng ta đã đụng vào bảng "submissions" để lấy điểm cao nhất và trạng thái tốt nhất
        if user_id:
            select_clause = """
                p.id, p.name, p.source, p.author_id, p.is_public, p.request_status, p.created_at,
                (SELECT MAX(s.score) FROM submissions s WHERE s.problem_id = p.id AND s.user_id = ?) as best_score,
                (SELECT s.status FROM submissions s WHERE s.problem_id = p.id AND s.user_id = ? ORDER BY s.score DESC, s.created_at DESC LIMIT 1) as best_status
            """
        else:
            select_clause = """
                p.id, p.name, p.source, p.author_id, p.is_public, p.request_status, p.created_at,
                NULL as best_score,
                NULL as best_status
            """

        if filter_mode == "public":
            if user_id:
                cursor.execute(f"""
                    SELECT {select_clause}
                    FROM problems p
                    WHERE p.is_public = 1 AND p.request_status IN ('APPROVED', 'NONE')
                    ORDER BY p.created_at DESC
                """, (user_id, user_id))
            else:
                cursor.execute(f"""
                    SELECT {select_clause}
                    FROM problems p
                    WHERE p.is_public = 1 AND p.request_status IN ('APPROVED', 'NONE')
                    ORDER BY p.created_at DESC
                """)
        elif filter_mode == "private":
                    if not user_id:
                        raise HTTPException(status_code=400, detail="user_id required for private filter")
                    # SỬA ĐIỀU KIỆN WHERE: Chỉ lọc theo p.author_id = ? thay vì cả p.is_public = 0
                    cursor.execute(f"""
                        SELECT {select_clause}
                        FROM problems p
                        WHERE p.author_id = ?
                        ORDER BY p.created_at DESC
                    """, (user_id, user_id, user_id))
        elif filter_mode == "all":
            if not user_id:
                raise HTTPException(status_code=400, detail="user_id required for all filter")
            cursor.execute(f"""
                SELECT {select_clause}
                FROM problems p
                WHERE p.is_public = 1 OR p.author_id = ?
                ORDER BY p.created_at DESC
            """, (user_id, user_id, user_id))
        else:
            raise HTTPException(status_code=400, detail="Invalid filter_mode. Use: public, private, or all")
        
        data = [dict(row) for row in cursor.fetchall()]
        return {"status": "success", "data": data}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

# Usecase: Get problem content to display in livecoding page
@app.get("/api/problems/{problem_id}/content")
def get_problem_content(problem_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, name, statement_path, theory_path, tutorial_path, solution_path, coding_path
            FROM problems
            WHERE id = ?
        """, (problem_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Problem not found")

        def read_file(path_value: Optional[str]) -> str:
            if not path_value:
                return ""
            if not os.path.exists(path_value):
                return ""
            with open(path_value, "r", encoding="utf-8") as f:
                return f.read()

        data = dict(row)
        
        form = {
            "status": "success",
            "data": {
                "id": data["id"],
                "name": data["name"],
                "statement_markdown": read_file(data["statement_path"]),
                "theory_markdown": read_file(data["theory_path"]),
                "tutorial_markdown": read_file(data["tutorial_path"]),
                "solution_markdown": read_file(data["solution_path"]),
                "coding_markdown": read_file(data["coding_path"]),
            }
        }

        return form
    except sqlite3.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database query error: {str(e)}"
        )

# Usecase: Run code on the first test case (for quick feedback in livecoding)
@app.post("/api/problems/{problem_id}/run")
async def run_problem_code(
    problem_id: int,
    payload: RunRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    submitted_code = payload.submitted_code
    cursor = db.cursor()
    
    # 1. Truy vấn đường dẫn từ Database
    cursor.execute("""
        SELECT input_folder_path, output_folder_path 
        FROM problems 
        WHERE id = ?
    """, (problem_id,))
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Problem not found")
        
    db_input_folder, db_output_folder = row[0], row[1]

    input_folder = os.path.abspath(db_input_folder) if db_input_folder else None
    output_folder = os.path.abspath(db_output_folder) if db_output_folder else None
    
    if not input_folder or not os.path.exists(input_folder):
        raise HTTPException(status_code=400, detail="Problem inputs are missing on the server")

    # 2. Tìm tệp tin Input Test Case đầu tiên
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])
    if not input_files:
        raise HTTPException(status_code=400, detail="No input JSON files found in testcases")
        
    first_input_file = input_files[0]
    input_file_path = os.path.join(input_folder, first_input_file)
    
    # Lấy đường dẫn tuyệt đối đến tệp tin input gốc
    abs_input_file_path = os.path.abspath(input_file_path)
    
    # 3. Tìm tệp tin Output tương ứng (Đáp án gốc để so sánh - Chế độ Read-only)
    output_file_path = os.path.join(output_folder, first_input_file)
    if not os.path.exists(output_file_path):
        output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.json')])
        if not output_files:
            raise HTTPException(status_code=400, detail="No output JSON files found in testcases")
        output_file_path = os.path.join(output_folder, output_files[0])
        
    abs_output_file_path = os.path.abspath(output_file_path)

    # 4. TẠO FILE OUTPUT TẠM THỜI (Bảo mật nâng cao)
    # Không cho phép người dùng ghi đè lên tệp đáp án gốc trên máy chủ
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_out:
        temp_output_path = tmp_out.name
    abs_temp_output_path = os.path.abspath(temp_output_path)

    # 5. Thay thế các chuỗi đường dẫn trong code bằng đường dẫn tuyệt đối
    # Sử dụng .replace('\\', '\\\\') để tránh lỗi escape ký tự gạch chéo ngược trên hệ điều hành Windows
    escaped_input_path = abs_input_file_path.replace('\\', '\\\\')
    escaped_output_path = abs_temp_output_path.replace('\\', '\\\\')
    
    print(escaped_input_path)
    submitted_code = submitted_code.replace("input.json", escaped_input_path)
    submitted_code = submitted_code.replace("output.json", escaped_output_path)

    # 6. Tạo file mã nguồn tạm thời để chạy
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as temp_file:
        temp_file.write(submitted_code)
        temp_file_path = temp_file.name

    # 7. Thực thi tiến trình độc lập (Subprocess) với Timeout giới hạn
    start_time = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=5.0  # Giới hạn tối đa 5 giây
        )
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    except subprocess.TimeoutExpired:
        if os.path.exists(abs_temp_output_path):
            os.remove(abs_temp_output_path)
        return {
            "status": "Time Limit Exceeded",
            "message": "Your program exceeded the 5.0s time limit.",
            "output": None,
            "expected": None,
            "elapsed_ms": 5000
        }
    except Exception as e:
        if os.path.exists(abs_temp_output_path):
            os.remove(abs_temp_output_path)
        return {
            "status": "Runtime Error",
            "message": f"Execution system failed: {str(e)}",
            "output": None,
            "expected": None,
            "elapsed_ms": 0
        }
    finally:
        # Luôn dọn dẹp file mã nguồn tạm thời sau khi kết thúc tiến trình
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # 8. Xử lý lỗi Runtime từ mã nguồn của người dùng
    if process.returncode != 0:
        if os.path.exists(abs_temp_output_path):
            os.remove(abs_temp_output_path)
        return {
            "status": "Runtime Error",
            "message": process.stderr.strip() or "Process exited with non-zero status.",
            "output": None,
            "expected": None,
            "elapsed_ms": elapsed_ms
        }

    # 9. Đọc dữ liệu đầu ra từ FILE TẠM THỜI mà chương trình người dùng vừa ghi xuống
    try:
        if not os.path.exists(abs_temp_output_path) or os.path.getsize(abs_temp_output_path) == 0:
            raise ValueError("Output file is empty or was not created.")
            
        with open(abs_temp_output_path, "r", encoding="utf-8") as f:
            user_output_json = json.load(f)
    except Exception as e:
        return {
            "status": "Presentation Error",
            "message": f"Your program failed to write valid JSON to output.json: {str(e)}",
            "output": process.stdout.strip()[:1000] if process.stdout else None,
            "expected": None,
            "elapsed_ms": elapsed_ms
        }
    finally:
        # Xóa file output tạm ngay lập tức sau khi đã đọc xong dữ liệu
        if os.path.exists(abs_temp_output_path):
            os.remove(abs_temp_output_path)

    # 10. Đọc tệp tin kết quả mong đợi gốc từ máy chủ (Read-only)
    try:
        with open(abs_output_file_path, "r", encoding="utf-8") as f:
            expected_output_json = json.load(f)
            
    except Exception as e:
        return {
            "status": "System Error",
            "message": f"Failed to load expected output file: {str(e)}",
            "output": user_output_json,
            "expected": None,
            "elapsed_ms": elapsed_ms
        }

    # 11. Thực hiện so sánh cấu trúc JSON động có sai số float
    if compare_approx(user_output_json, expected_output_json):
        return {
            "status": "Accepted",
            "message": "Your code passed the first testcase successfully!",
            "output": user_output_json,
            "expected": expected_output_json,
            "elapsed_ms": elapsed_ms
        }
    else:
        return {
            "status": "Wrong Answer",
            "message": "Your output does not match the expected result.",
            "output": user_output_json,
            "expected": expected_output_json,
            "elapsed_ms": elapsed_ms
        }

# Usecase : Submit code to run on all test cases and save results to database
@app.post("/api/problems/{problem_id}/submit")
async def submit_problem_code(
    problem_id: int,
    payload: SubmitRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    user_id = payload.user_id
    submitted_code = payload.submitted_code
    cursor = db.cursor()

    # 1. Lấy thông tin thư mục chứa toàn bộ Test Case của bài tập
    cursor.execute("""
        SELECT input_folder_path, output_folder_path 
        FROM problems 
        WHERE id = ?
    """, (problem_id,))
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Problem not found")

    db_input_folder, db_output_folder = row[0], row[1]
    
    input_folder = os.path.abspath(db_input_folder) if db_input_folder else None
    output_folder = os.path.abspath(db_output_folder) if db_output_folder else None

    if not input_folder or not os.path.exists(input_folder):
        raise HTTPException(status_code=400, detail="Problem inputs are missing on the server")

    # 2. Quét toàn bộ danh sách tệp tin Input Test Case (.json) và sắp xếp theo thứ tự
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])
    if not input_files:
        raise HTTPException(status_code=400, detail="No input JSON files found for evaluation")

    total_tc = len(input_files)
    passed_tc = 0
    
    # Định nghĩa các hằng số trạng thái để đồng bộ
    db_status = "accepted"  # Trạng thái lưu vào Database (lowercase)
    results: List[Dict[str, Any]] = []

    # 3. Chạy vòng lặp kiểm thử từng Test Case độc lập
    for idx, in_file in enumerate(input_files):
        # Đường dẫn tuyệt đối đến tệp tin input gốc của Test Case này
        abs_in_path = os.path.abspath(os.path.join(input_folder, in_file))

        # Tìm tệp tin Output đáp án tương ứng
        abs_out_path = os.path.abspath(os.path.join(output_folder, in_file))
        if not os.path.exists(abs_out_path):
            output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.json')])
            if len(output_files) > idx:
                abs_out_path = os.path.abspath(os.path.join(output_folder, output_files[idx]))
            else:
                # Nếu thiếu file đáp án trên máy chủ
                results.append({
                    "testcase": in_file,
                    "status": "System Error",
                    "user_output": "Missing expected output on server"
                })
                db_status = "wrong_answer" if db_status == "accepted" else db_status
                continue

        # Tạo đường dẫn đầu ra tạm thời an toàn để người dùng ghi kết quả xuống
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_out:
            temp_output_path = tmp_out.name
        abs_temp_output_path = os.path.abspath(temp_output_path)

        # Thay thế "input.json" và "output.json" bằng đường dẫn tuyệt đối tương ứng
        escaped_input_path = abs_in_path.replace('\\', '\\\\')
        escaped_output_path = abs_temp_output_path.replace('\\', '\\\\')
        
        tc_submitted_code = submitted_code.replace("input.json", escaped_input_path)
        tc_submitted_code = tc_submitted_code.replace("output.json", escaped_output_path)

        # Tạo file code tạm thời để chạy độc lập
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(tc_submitted_code)
            temp_file_path = temp_file.name

        # Thực thi tiến trình của Test Case hiện tại
        tc_status = "Wrong Answer"
        user_output_display = ""
        
        try:
            process = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=5.0  # Mỗi testcase chạy tối đa 5 giây
            )
            
            # Xử lý lỗi Runtime của mã nguồn người dùng
            if process.returncode != 0:
                tc_status = "Runtime Error"
                user_output_display = process.stderr.strip()[:150]
                if db_status in ("accepted", "wrong_answer"):
                    db_status = "runtime_error"
            else:
                # Đọc dữ liệu đầu ra mà chương trình ghi xuống file tạm
                try:
                    if not os.path.exists(abs_temp_output_path) or os.path.getsize(abs_temp_output_path) == 0:
                        raise ValueError("No output generated")
                        
                    with open(abs_temp_output_path, "r", encoding="utf-8") as f:
                        user_output_json = json.load(f)
                    
                    user_output_display = str(user_output_json)[:150] # Cắt bớt độ dài chuỗi để hiển thị gọn
                    
                    # Đọc kết quả đáp án chuẩn để so sánh
                    with open(abs_out_path, "r", encoding="utf-8") as f:
                        expected_output_json = json.load(f)

                    # Tiến hành so sánh cấu trúc
                    if compare_approx(user_output_json, expected_output_json):
                        tc_status = "Accepted"
                        passed_tc += 1
                    else:
                        tc_status = "Wrong Answer"
                        if db_status == "accepted":
                            db_status = "wrong_answer"
                except Exception as e:
                    tc_status = "Wrong Answer"
                    user_output_display = f"Format Error: {str(e)}"
                    if db_status == "accepted":
                        db_status = "wrong_answer"

        except subprocess.TimeoutExpired:
            tc_status = "Time Limit Exceeded"
            user_output_display = "TLE (5.0s limit)"
            if db_status != "runtime_error":
                db_status = "time_limit_exceeded"
        except Exception as e:
            tc_status = "Runtime Error"
            user_output_display = f"Execution system error: {str(e)}"
            if db_status in ("accepted", "wrong_answer"):
                db_status = "runtime_error"
        finally:
            # Luôn giải phóng tài nguyên tạm thời của Test Case này
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(abs_temp_output_path):
                os.remove(abs_temp_output_path)

        # Lưu kết quả kiểm thử của Test Case này vào mảng kết quả
        results.append({
            "testcase": in_file,
            "status": tc_status,
            "user_output": user_output_display
        })

    # 4. Tính toán điểm số cuối cùng (thang điểm 100)
    score = int((passed_tc / total_tc) * 100) if total_tc > 0 else 0

    # Chuyển đổi trạng thái chung sang dạng chữ viết hoa (Friendly Name) để trả về Frontend
    status_mapping = {
        "accepted": "Accepted",
        "wrong_answer": "Wrong Answer",
        "runtime_error": "Runtime Error",
        "time_limit_exceeded": "Time Limit Exceeded"
    }
    friendly_status = status_mapping.get(db_status, "Wrong Answer")

    # 5. Lưu bản ghi nộp bài (submission) vào SQLite Database
    test_results_json_str = json.dumps(results)
    
    try:
        cursor.execute("""
            INSERT INTO submissions (user_id, problem_id, submitted_code, status, score, test_results)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            problem_id,
            submitted_code,
            db_status, # Lưu trạng thái chuẩn (lowercase) vào DB để khớp với các kiểm tra khác
            score,
            test_results_json_str
        ))
        db.commit()
        submission_id = cursor.lastrowid
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database write error: {str(e)}")

    # 6. Trả về đúng định dạng dữ liệu Frontend yêu cầu hiển thị
    return {
        "status": friendly_status,
        "score": score,
        "submission_id": submission_id,
        "results": results
    }

# Usecase : Fetch submission history for a problem (for livecoding session review)
@app.get("/api/problems/{problem_id}/submissions")
def get_problem_submissions(
    problem_id: int, 
    user_id: int, 
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, user_id, problem_id, submitted_code, status, score, test_results, created_at
            FROM submissions
            WHERE problem_id = ? AND user_id = ?
            ORDER BY created_at DESC
        """, (problem_id, user_id))
        
        rows = cursor.fetchall()
        submissions_list = []
        for row in rows:
            sub_dict = dict(row)
            # Khôi phục chuỗi JSON của test_results thành mảng/đối tượng Python
            if sub_dict.get("test_results"):
                try:
                    sub_dict["test_results"] = json.loads(sub_dict["test_results"])
                except Exception:
                    sub_dict["test_results"] = []
            submissions_list.append(sub_dict)
            
        return {"status": "success", "data": submissions_list}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

def fetch_user_profile_record(cursor: sqlite3.Cursor, user_id: int) -> dict:
    cursor.execute("""
        SELECT id, username, display_name, email, role, avatar_url, created_at, status
        FROM users
        WHERE id = ?
    """, (user_id,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(user)

def get_authenticated_identity(authorization: Optional[str]) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authentication header")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Authentication token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return {
        "username": username,
        "user_id": payload.get("user_id"),
    }

def require_account_owner(cursor: sqlite3.Cursor, user_id: int, authorization: Optional[str]) -> dict:
    user = fetch_user_profile_record(cursor, user_id)
    authenticated_identity = get_authenticated_identity(authorization)
    authenticated_user_id = authenticated_identity.get("user_id")
    authenticated_username = authenticated_identity.get("username", "")

    if authenticated_user_id is not None:
        try:
            if int(authenticated_user_id) == int(user_id):
                return user
        except (TypeError, ValueError):
            pass

    if authenticated_username.lower() != user["username"].lower():
        raise HTTPException(status_code=403, detail="Cannot update another user's account")
    return user

# Usecase : Get user profile information (for profile page and admin management)
@app.get("/api/users/profile/{user_id}")
def get_user_profile(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        user = fetch_user_profile_record(cursor, user_id)
        return {"status": "success", "data": user}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

# Usecase : Get user profile by id (for account update flow)
@app.get("/api/users/{user_id}")
def get_user_profile_by_id(
    user_id: int,
    authorization: Optional[str] = Header(None),
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    try:
        user = require_account_owner(cursor, user_id, authorization)
        return {"status": "success", "data": user}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

# Usecase : Update user profile information
@app.put("/api/users/{user_id}")
def update_user_profile(
    user_id: int,
    username: Optional[str] = Form(None),
    display_name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    avatar: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(None),
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    try:
        user = require_account_owner(cursor, user_id, authorization)

        update_fields = []
        params = []
        missing_fields = []
        effective_display_name = user["display_name"]

        if display_name is not None:
            effective_display_name = display_name.strip()

        if username is not None:
            username = username.strip()
            if not username:
                missing_fields.append("username")
            else:
                username_errors = validate_username(username)
                if username_errors:
                    raise HTTPException(status_code=400, detail=" ".join(username_errors))

                cursor.execute(
                    "SELECT 1 FROM users WHERE id != ? AND LOWER(username) = LOWER(?)",
                    (user_id, username)
                )
                if cursor.fetchone():
                    raise HTTPException(status_code=400, detail="Username already exists")

                if effective_display_name and username.lower() == effective_display_name.lower():
                    raise HTTPException(
                        status_code=400,
                        detail="Username cannot duplicate a display name"
                    )

                cursor.execute(
                    "SELECT 1 FROM users WHERE id != ? AND LOWER(display_name) = LOWER(?)",
                    (user_id, username)
                )
                if cursor.fetchone():
                    raise HTTPException(
                        status_code=400,
                        detail="Username already exists as a display name"
                    )

                update_fields.append("username = ?")
                params.append(username)

        if display_name is not None:
            display_name = display_name.strip()
            if not display_name:
                missing_fields.append("display_name")
            else:
                update_fields.append("display_name = ?")
                params.append(display_name)

        if email is not None:
            email = email.strip()
            if not email:
                missing_fields.append("email")
            else:
                email_error = validate_email_format(email)
                if email_error:
                    raise HTTPException(status_code=400, detail=email_error)

                cursor.execute(
                    "SELECT 1 FROM users WHERE id != ? AND LOWER(email) = LOWER(?)",
                    (user_id, email)
                )
                if cursor.fetchone():
                    raise HTTPException(status_code=400, detail="Email already exists")

                update_fields.append("email = ?")
                params.append(email)

        if password is not None:
            if not password or not password.strip():
                missing_fields.append("password")
            else:
                password_errors = validate_password(password)
                if password_errors:
                    raise HTTPException(status_code=400, detail=" ".join(password_errors))
                update_fields.append("password_hash = ?")
                params.append(hash_password(password))

        # Handle avatar file upload
        if avatar is not None:
            # Validate content type
            allowed_types = {"image/jpeg", "image/png", "image/webp"}
            if avatar.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail="Avatar must be JPG, PNG, or WEBP image")
            # Read file bytes to check size (max 5MB)
            file_bytes = avatar.file.read()
            max_size = 5 * 1024 * 1024
            if len(file_bytes) > max_size:
                raise HTTPException(status_code=400, detail="Avatar file size must be <= 5MB")
            # Determine file extension
            ext_map = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
            ext = ext_map.get(avatar.content_type, "jpg")
            # Generate unique filename
            filename = f"{uuid.uuid4()}.{ext}"
            avatar_path = os.path.join("storage", "avatars", filename)
            # Write file to disk
            with open(avatar_path, "wb") as out_file:
                out_file.write(file_bytes)
            # Set avatar URL (served via /avatars mount)
            avatar_url = f"/avatars/{filename}"
            update_fields.append("avatar_url = ?")
            params.append(avatar_url) 
            # Reset file cursor for potential later use
            avatar.file.seek(0)


        if missing_fields:
            missing_text = ", ".join(missing_fields)
            raise HTTPException(status_code=400, detail=f"Missing required fields: {missing_text}")

        if not update_fields:
            raise HTTPException(status_code=400, detail="No profile fields provided for update")

        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        cursor.execute(
            f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?",
            params + [user_id]
        )
        db.commit()
        updated_user = fetch_user_profile_record(cursor, user_id)
        access_token = create_access_token(
            data={
                "sub": updated_user["username"],
                "role": updated_user["role"],
                "user_id": updated_user["id"],
            },
            expires_delta=datetime.timedelta(minutes=30)
        )
        return {
            "status": "success",
            "message": "Update successful",
            "access_token": access_token,
            "token_type": "bearer",
            "data": updated_user,
        }
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database execution error: {str(e)}")

# Usecase : Deactivate user account and delete associated data
@app.post("/api/users/{user_id}/deactivate")
def deactivate_user_account(
    user_id: int,
    payload: AccountDeactivationRequest,
    authorization: Optional[str] = Header(None),
    db: sqlite3.Connection = Depends(get_db)
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Account deactivation requires confirmation")

    cursor = db.cursor()
    try:
        require_account_owner(cursor, user_id, authorization)

        cursor.execute("SELECT id FROM problems WHERE author_id = ?", (user_id,))
        problem_rows = cursor.fetchall()
        problem_ids = [row["id"] for row in problem_rows]

        if problem_ids:
            placeholders = ", ".join(["?"] * len(problem_ids))
            cursor.execute(
                f"DELETE FROM roadmap_problems WHERE problem_id IN ({placeholders})",
                problem_ids
            )

        cursor.execute("DELETE FROM submissions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM refresh_tokens WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM draft_problem_sessions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM roadmaps WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM problems WHERE author_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))

        db.commit()
        return {
            "status": "success",
            "message": "Account deactivated and data deleted"
        }
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database execution error: {str(e)}")

# Usecase : Add admin role to a user (for admin management)
@app.post("/api/users/make-admin")
def make_user_admin(payload: MakeAdminRequest, db: sqlite3.Connection = Depends(get_db)):
    admin_secret = os.getenv("ADMIN_SECRET_KEY", "SUPER_SECRET_ADMIN_2026")
    print(admin_secret)
    if payload.secret_key != admin_secret:
        raise HTTPException(status_code=400, detail="Invalid admin secret key. Access denied.")
        
    cursor = db.cursor()
    try:
        # Cập nhật quyền lên 'admin'
        cursor.execute("UPDATE users SET role = 'admin' WHERE id = ?", (payload.user_id,))
        db.commit()
        return {
            "status": "success", 
            "message": "Congratulations! You have successfully upgraded to Admin. Please re-login to apply changes."
        }
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database execution error: {str(e)}")

# Usecase : Problem approval workflow (For user to request approval and admin to approve/reject)
@app.post("/api/problems/{problem_id}/request-approval")
def request_problem_approval(problem_id: int, payload: RequestApprovalPayload, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # Kiểm tra xem bài tập có tồn tại và người yêu cầu có đúng là tác giả không
        cursor.execute("SELECT author_id, request_status, is_public FROM problems WHERE id = ?", (problem_id,))
        problem = cursor.fetchone()
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        if problem["author_id"] != payload.user_id:
            raise HTTPException(status_code=403, detail="Only the problem author can request public approval")
        
        if problem["is_public"] == 1:
            raise HTTPException(status_code=400, detail="Problem is already public")
        
        if problem["request_status"] == "PENDING":
            raise HTTPException(status_code=400, detail="Approval already requested and is pending")
        
        cursor.execute("UPDATE problems SET request_status = 'PENDING' WHERE id = ?", (problem_id,))
        db.commit()
        return {"status": "success", "message": "Approval request submitted successfully"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# 2. Admin phê duyệt bài tập lên hệ thống (Chuyển is_public = 1)
@app.post("/api/problems/{problem_id}/approve")
def approve_problem(problem_id: int, payload: AdminActionPayload, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # Kiểm tra bảo mật xem người thực thi có phải admin thật không
        cursor.execute("SELECT role FROM users WHERE id = ?", (payload.admin_id,))
        user = cursor.fetchone()
        if not user or user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Unauthorized. Admin privileges required.")
        
        cursor.execute("SELECT id FROM problems WHERE id = ?", (problem_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Problem not found")
        
        cursor.execute("UPDATE problems SET request_status = 'APPROVED', is_public = 1 WHERE id = ?", (problem_id,))
        db.commit()
        return {"status": "success", "message": "Problem approved and published successfully"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# 3. Admin từ chối phê duyệt bài tập (Giữ nguyên is_public = 0)
@app.post("/api/problems/{problem_id}/reject")
def reject_problem(problem_id: int, payload: AdminActionPayload, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # Kiểm tra bảo mật Admin
        cursor.execute("SELECT role FROM users WHERE id = ?", (payload.admin_id,))
        user = cursor.fetchone()
        if not user or user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Unauthorized. Admin privileges required.")
        
        cursor.execute("UPDATE problems SET request_status = 'REJECTED', is_public = 0 WHERE id = ?", (problem_id,))
        db.commit()
        return {"status": "success", "message": "Problem approval request rejected"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# 4. Admin lấy danh sách toàn bộ các bài tập đang chờ duyệt (Pending Queue)
@app.get("/api/problems/pending-requests")
def get_pending_requests(admin_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("SELECT role FROM users WHERE id = ?", (admin_id,))
        user = cursor.fetchone()
        if not user or user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        cursor.execute("""
            SELECT p.id, p.name, p.source, p.author_id, p.is_public, p.request_status, p.created_at,
                   u.display_name as author_name
            FROM problems p
            JOIN users u ON u.id = p.author_id
            WHERE p.request_status = 'PENDING'
            ORDER BY p.created_at ASC
        """)
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")


# 5. User lấy danh sách lịch sử yêu cầu phê duyệt của chính họ
@app.get("/api/problems/my-requests")
def get_my_requests(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, name, source, is_public, request_status, created_at
            FROM problems
            WHERE author_id = ? AND request_status IN ('PENDING', 'REJECTED', 'APPROVED')
            ORDER BY created_at DESC
        """, (user_id,))
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

# Usecase: Generate proposed problems from GitHub repository using AI (DeepWiki)
@app.post("/api/problems/problems_from_repo", response_model=ProblemsFromRepoResponse)
def get_problems_from_repo(payload: ProblemsFromRepo, db: sqlite3.Connection = Depends(get_db)):
    try:
        owner, repo = get_repo_owner_and_name(payload.repository_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cache_data = check_wiki_cache(owner, repo)  
    if not cache_data: 
        print(f"[Warning] Repo {owner}/{repo} wasn't cached. DeepWiki will take more time to investigate.")

    try:
        question_prompt = get_problems_from_repo_prompt.format(
            repository_url=payload.repository_url,
            level=payload.level,
            framework=payload.framework or 'Auto-detect',
            user_note=payload.user_note or 'None'
        )

        ai_raw_response = ask_question(payload.repository_url, question_prompt)
        json_string = clean_json_response(ai_raw_response)
        parsed_problems = [ProposedProblem(**item) for item in json.loads(json_string)]

        cursor = db.cursor()
        # Thêm lưu num_test_cases vào cột DB tương ứng
        cursor.execute("""
            INSERT INTO draft_problem_sessions (roadmap_name, repository_url, user_id, problems_json, num_test_cases, status)
            VALUES (?, ?, ?, ?, ?, 'draft')
        """, (payload.roadmap_name, payload.repository_url, payload.user_id, json_string, payload.num_test_cases))
        
        db.commit()
        session_id = cursor.lastrowid
        
        return ProblemsFromRepoResponse(
            status="success",
            message="Generated proposed problem list successfully.",
            data=ProblemsFromRepoDataResponse(
                session_id=session_id,
                repository_url=payload.repository_url,
                roadmap_name=payload.roadmap_name,
                proposed_problems=parsed_problems
            )
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except json.JSONDecodeError as e:
        print(f"[Debug] AI Response failed to parse to JSON:\n{ai_raw_response}")
        raise HTTPException(
            status_code=502, 
            detail="AI generated an invalid format. Please try again with different notes."
        )
    except ValidationError as e:
        print(f"[Debug] Pydantic Validation Error:\n{e.json()}")
        raise HTTPException(
            status_code=502,
            detail="AI generated valid JSON but it did not match the required schema structure."
        )
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Database error during session creation: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/api/problems/draft_sessions")
def get_draft_sessions(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, roadmap_name, repository_url, status, num_test_cases, created_at 
            FROM draft_problem_sessions 
            WHERE user_id = ? AND status = 'draft'
            ORDER BY id DESC
        """, (user_id,))
        rows = cursor.fetchall()
        return {"status": "success", "data": [dict(row) for row in rows]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.post("/api/problems/draft_sessions/finalize")
def finalize_session(payload: FinalizeSessionRequest, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT repository_url, user_id, problems_json, num_test_cases, status 
            FROM draft_problem_sessions 
            WHERE id = ?
        """, (payload.session_id,))
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["status"] == "finalized":
            raise HTTPException(status_code=400, detail="Session already finalized")
            
        repo_url = session["repository_url"]
        user_id = session["user_id"]
        num_test_cases = session["num_test_cases"]
        problems_list = json.loads(session["problems_json"])
        
        try:
            # Tạo lộ trình chính thức
            cursor.execute("""
                INSERT INTO roadmaps (user_id, name, repository_url, level, num_test_cases, status)
                VALUES (?, ?, ?, ?, ?, 'draft')
            """, (user_id, payload.roadmap_title, repo_url, "medium", num_test_cases))
            roadmap_id = cursor.lastrowid
            
            # Chỉ tạo các bước nháp trong roadmap_problems, không tạo bản ghi rỗng ở bảng problems
            created_steps = []
            for index, prob in enumerate(problems_list, start=1):
                name = prob["title"].strip()
                description = prob.get("description", "").strip()
                target_module = prob.get("target_module", "").strip()
                
                cursor.execute("""
                    INSERT INTO roadmap_problems (roadmap_id, name, description, target_module, order_index, status)
                    VALUES (?, ?, ?, ?, ?, 'pending')
                """, (roadmap_id, name, description, target_module, index))
                step_id = cursor.lastrowid
                
                created_steps.append({
                    "step_id": step_id,
                    "name": name,
                    "order_index": index,
                    "status": "pending"
                })
                
            cursor.execute("UPDATE draft_problem_sessions SET status = 'finalized' WHERE id = ?", (payload.session_id,))
            db.commit()
            
            return {
                "status": "success", 
                "message": "Roadmap created successfully. Steps initialized as drafts.",
                "data": {
                    "roadmap_id": roadmap_id,
                    "name": payload.roadmap_title,
                    "repository_url": repo_url,
                    "steps": created_steps
                }
            }
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Transaction error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Finalize failed: {str(e)}")


app.post("/api/problems/draft_sessions/feedback")
def update_draft_session_with_feedback(payload: FeedbackRequest, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # 1. Đọc danh sách bài tập cũ từ session_id
        cursor.execute("SELECT repository_url, problems_json FROM draft_problem_sessions WHERE id = ?", (payload.session_id,))
        session = cursor.fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        current_problems = session["problems_json"]
        repo_url = session["repository_url"]
        
        # 2. Xây dựng prompt yêu cầu AI chỉnh sửa danh sách dựa trên feedback của người dùng
        question_prompt = feedback_prompt.format(
            repo_url=repo_url,
            current_problems=current_problems,
            payload=payload
        )
        
        ai_response = ask_question(repo_url, question_prompt)
        print(f"\n[DEBUG] AI Response from DeepWiki/Groq:\n{ai_response}\n")
        json_string = clean_json_response(ai_response)
        
        # Validate định dạng JSON mới
        parsed_problems = [ProposedProblem(**item) for item in json.loads(json_string)]
        
        # 3. Cập nhật đè danh sách mới vào database nháp
        cursor.execute("""
            UPDATE draft_problem_sessions 
            SET problems_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (json_string, payload.session_id))
        db.commit()
        
        return {
            "status": "success",
            "message": "Updated draft problems with user feedback successfully.",
            "data": {
                "session_id": payload.session_id,
                "proposed_problems": parsed_problems
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/problems/draft_sessions/finalize")
def finalize_session(payload: FinalizeSessionRequest, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # 1. Đọc thông tin phiên nháp bao gồm cả số lượng testcases
        cursor.execute("""
            SELECT repository_url, user_id, problems_json, num_test_cases, status 
            FROM draft_problem_sessions 
            WHERE id = ?
        """, (payload.session_id,))
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["status"] == "finalized":
            raise HTTPException(status_code=400, detail="Session already finalized")
            
        repo_url = session["repository_url"]
        user_id = session["user_id"]
        num_test_cases = session["num_test_cases"]
        problems_list = json.loads(session["problems_json"])
        
        try:
            # 2. Tạo lộ trình chính thức, kế thừa trường num_test_cases
            cursor.execute("""
                INSERT INTO roadmaps (user_id, name, repository_url, level, num_test_cases, status)
                VALUES (?, ?, ?, ?, ?, 'draft')
            """, (user_id, payload.roadmap_title, repo_url, "medium", num_test_cases))
            roadmap_id = cursor.lastrowid
            
            # 3. Tạo các bài tập con dưới dạng bản nháp riêng tư trống
            created_problems = []
            for index, prob in enumerate(problems_list, start=1):
                name = prob["title"].strip()
                name_slug = normalize_problem_name(name)
                problem_folder = f"{storage_path}problems/{name_slug}"
                os.makedirs(problem_folder, exist_ok=True)
                
                # Tạo file statement.md giữ chỗ tạm thời
                statement_path = os.path.join(problem_folder, "statement.md")
                with open(statement_path, "w", encoding="utf-8") as f:
                    f.write(f"# {name}\n\n*Problem details are being initialized by AI. Please click 'Create Detailedly' to complete.*")
                
                # Thêm bản ghi bài tập mới vào bảng problems
                cursor.execute("""
                    INSERT INTO problems (name, source, statement_path, author_id, is_public, request_status)
                    VALUES (?, ?, ?, ?, 0, 'NONE')
                """, (name, f"Roadmap: {payload.roadmap_title}", statement_path, user_id))
                problem_id = cursor.lastrowid
                
                # Liên kết lộ trình
                cursor.execute("""
                    INSERT INTO roadmap_problems (roadmap_id, problem_id, name, order_index, status)
                    VALUES (?, ?, ?, ?, 'pending')
                """, (roadmap_id, problem_id, name, index))
                
                created_problems.append({
                    "problem_id": problem_id,
                    "name": name,
                    "order_index": index,
                    "statement_path": statement_path
                })
                
            # 4. Đánh dấu phiên nháp đã hoàn thành (finalized)
            cursor.execute("UPDATE draft_problem_sessions SET status = 'finalized' WHERE id = ?", (payload.session_id,))
            db.commit()
            
            return {
                "status": "success", 
                "message": "Roadmap created successfully. Problems initialized as private drafts.",
                "data": {
                    "roadmap_id": roadmap_id,
                    "name": payload.roadmap_title,
                    "repository_url": repo_url,
                    "problems": created_problems
                }
            }
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Transaction error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Finalize failed: {str(e)}")

@app.get("/api/roadmaps/{roadmap_id}")
def get_roadmap_detail(roadmap_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, name, repository_url, level, user_note, framework, num_test_cases, status, user_id, created_at 
            FROM roadmaps 
            WHERE id = ?
        """, (roadmap_id,))
        roadmap = cursor.fetchone()
        if not roadmap:
            raise HTTPException(status_code=404, detail="Roadmap not found")

        # LEFT JOIN sang bảng problems để bóc thông tin nếu trạng thái là 'saved'
        cursor.execute("""
            SELECT rp.id AS step_id, rp.problem_id, rp.name, rp.order_index, rp.description, rp.target_module, rp.status AS step_status,
                   p.statement_path, p.theory_path, p.tutorial_path, p.solution_path, p.coding_path, p.checker_path,
                   p.input_folder_path, p.output_folder_path
            FROM roadmap_problems rp
            LEFT JOIN problems p ON p.id = rp.problem_id
            WHERE rp.roadmap_id = ?
            ORDER BY rp.order_index ASC
        """, (roadmap_id,))
        steps_raw = [dict(row) for row in cursor.fetchall()]
        
        steps = []
        for step in steps_raw:
            step_dict = dict(step)
            if step_dict.get("step_status") == "saved":
                has_all_materials = all([
                    step_dict.get("statement_path"),
                    step_dict.get("theory_path"),
                    step_dict.get("tutorial_path"),
                    step_dict.get("solution_path"),
                    step_dict.get("coding_path"),
                    step_dict.get("checker_path")
                ])
                step_dict["has_materials"] = has_all_materials
            else:
                step_dict["has_materials"] = False
            steps.append(step_dict)

        data = dict(roadmap)
        data["problems"] = steps # Giữ nguyên trường 'problems' để không lỗi giao diện React
        return {"status": "success", "data": data}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.get("/api/roadmaps")
def list_roadmaps(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT r.id, r.name, r.repository_url, r.level, r.num_test_cases, r.status, r.created_at,
                   COUNT(rp.problem_id) AS problem_count
            FROM roadmaps r
            LEFT JOIN roadmap_problems rp ON rp.roadmap_id = r.id
            WHERE r.user_id = ?
            GROUP BY r.id
            ORDER BY r.id DESC
        """, (user_id,))
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
    
@app.post("/api/roadmap-steps/{step_id}/create_detailedly")
def create_problem_detailedly(step_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    
    # 1. Truy vấn thông tin của step nháp kèm cấu hình roadmap
    cursor.execute("""
        SELECT rp.name, r.repository_url, r.name AS roadmap_name, r.num_test_cases
        FROM roadmap_problems rp
        JOIN roadmaps r ON rp.roadmap_id = r.id
        WHERE rp.id = ?
    """, (step_id,))
    record = cursor.fetchone()
    
    if not record:
        raise HTTPException(status_code=404, detail="Step not found in active Roadmap.")
        
    name = record["name"]
    repo_url = record["repository_url"]
    roadmap_name = record["roadmap_name"]
    num_test_cases = record["num_test_cases"] or 3
    
    # Lưu thư mục tạm thời dạng nháp dựa theo step_id
    draft_folder = f"{storage_path}draft_problems/{step_id}"
    
    # 2. Tạo prompt truyền động cấu hình cho AI biên soạn
    try:
        question_prompt = create_detailedly_prompt.format(
            title=name,
            repository_url=repo_url,
            roadmap_title=roadmap_name,
            num_test_cases=num_test_cases
        )
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Prompt formatting error: {str(e)}")

    ai_raw_response = ask_question(repo_url, question_prompt)
    
    try:
        raw_json_data = parse_and_repair_json(ai_raw_response)
        raw_json_data = validate_and_ensure_complete_materials(raw_json_data)
        materials = DetailedProblemMaterials(**raw_json_data)
        
        # Đường dẫn tệp nháp tạm thời
        statement_path = os.path.join(draft_folder, "statement.md")
        theory_path = os.path.join(draft_folder, "theory.md")
        tutorial_path = os.path.join(draft_folder, "tutorial.md")
        solution_path = os.path.join(draft_folder, "solution.py")
        coding_path = os.path.join(draft_folder, "coding.py")
        checker_path = os.path.join(draft_folder, "checker.py")

        os.makedirs(draft_folder, exist_ok=True)
        
        with open(statement_path, "w", encoding="utf-8") as f:
            f.write(materials.statement)
        with open(theory_path, "w", encoding="utf-8") as f:
            f.write(materials.theory)
        with open(tutorial_path, "w", encoding="utf-8") as f:
            f.write(materials.tutorial)
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(extract_python_code(materials.solution))
        with open(coding_path, "w", encoding="utf-8") as f:
            f.write(extract_python_code(materials.coding))
        with open(checker_path, "w", encoding="utf-8") as f:
            f.write(extract_python_code(materials.checker))
        
        # =================== 4. PIPELINE TỰ ĐỘNG SINH DỮ LIỆU TEST CASES ĐỘNG ===================
        temp_input_dir = os.path.join(draft_folder, "inputs")
        temp_output_dir = os.path.join(draft_folder, "outputs")
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)

        try:
            test_inputs_list = json.loads(materials.test_inputs)
        except Exception:
            test_inputs_list = []

        if len(test_inputs_list) < num_test_cases:
            while len(test_inputs_list) < num_test_cases:
                test_inputs_list.append(test_inputs_list[-1] if test_inputs_list else {})

        # Tiến hành lưu tệp tin input và chạy solution.py để sinh tệp output chuẩn xác
        for i in range(1, num_test_cases + 1):
            inp = test_inputs_list[i - 1]
            inp_file_path = os.path.join(temp_input_dir, f"input_{i}.json")
            out_file_path = os.path.join(temp_output_dir, f"output_{i}.json")
            
            # Ghi file input JSON của testcase
            with open(inp_file_path, "w", encoding="utf-8") as f_in:
                json.dump(inp, f_in)
                
            # Tạo file input.json và output.json tạm thời ngay tại thư mục chứa để solution.py đọc/ghi tại chỗ
            temp_input_json = os.path.join(draft_folder, "input.json")
            temp_output_json = os.path.join(draft_folder, "output.json")
            
            with open(temp_input_json, "w", encoding="utf-8") as f_temp:
                json.dump(inp, f_temp)
                
            try:
                # Chạy solution.py từ đúng thư mục làm việc (cwd) của nó để đọc/ghi file chuẩn xác
                process = subprocess.run(
                    [sys.executable, "solution.py"],
                    cwd=draft_folder,
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                
                if process.returncode == 0 and os.path.exists(temp_output_json):
                    # Sao chép file output.json tạm thời vừa sinh ra thành file output_{i}.json chính thức
                    shutil.copy2(temp_output_json, out_file_path)
                else:
                    print(f"[ERROR] Generator failed for testcase {i}: {process.stderr.strip() if process.stderr else 'No stderr'}")
                    with open(out_file_path, "w", encoding="utf-8") as f_out:
                        json.dump({"error": f"Execution failed: {process.stderr.strip()[:100] if process.stderr else 'unknown'}"}, f_out)
            except Exception as e:
                print(f"[ERROR] Exception during testcase generation: {str(e)}")
                with open(out_file_path, "w", encoding="utf-8") as f_out:
                    json.dump({"error": str(e)}, f_out)
            finally:
                # Giải phóng các file input/output tạm thời sau mỗi lượt chạy để tránh nhầm lẫn dữ liệu
                if os.path.exists(temp_input_json):
                    os.remove(temp_input_json)
                if os.path.exists(temp_output_json):
                    os.remove(temp_output_json)

        # Nén ZIP nháp
        input_zip_path = os.path.join(draft_folder, "input.zip")
        output_zip_path = os.path.join(draft_folder, "output.zip")

        with zipfile.ZipFile(input_zip_path, 'w') as zip_in:
            for file in sorted(os.listdir(temp_input_dir)):
                zip_in.write(os.path.join(temp_input_dir, file), arcname=file)

        with zipfile.ZipFile(output_zip_path, 'w') as zip_out:
            for file in sorted(os.listdir(temp_output_dir)):
                zip_out.write(os.path.join(temp_output_dir, file), arcname=file)

        # Cập nhật trạng thái sang "generated" (đã sinh tài liệu nháp thành công)
        cursor.execute("UPDATE roadmap_problems SET status = 'generated' WHERE id = ?", (step_id,))
        db.commit()
        
        return {
            "status": "success",
            "message": f"Successfully generated draft materials.",
            "data": {
                "step_id": step_id,
                "status": "generated"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Draft generation failed: {str(e)}")
    
    
@app.post("/api/roadmap-steps/{step_id}/save_to_problem")
def save_step_to_problem(step_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # 1. Truy cập thông tin bước nháp
        cursor.execute("""
            SELECT rp.name, rp.roadmap_id, r.user_id, rp.status
            FROM roadmap_problems rp
            JOIN roadmaps r ON rp.roadmap_id = r.id
            WHERE rp.id = ?
        """, (step_id,))
        record = cursor.fetchone()
        
        if not record:
            raise HTTPException(status_code=404, detail="Step not found.")
        if record["status"] != "generated":
            raise HTTPException(status_code=400, detail="Please click 'Create Detailedly' first before saving.")
            
        name = record["name"]
        user_id = record["user_id"]
        roadmap_id = record["roadmap_id"]
        name_slug = normalize_problem_name(name)
        
        # Đảm bảo tên bài tập không bị trùng lặp trong hệ thống bài tập chung
        cursor.execute("SELECT id FROM problems WHERE name = ?", (name_slug,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="The problem name already exists. Please delete it or rename first.")
            
        draft_folder = f"{storage_path}draft_problems/{step_id}"
        official_folder = f"{storage_path}problems/{name_slug}"
        
        if not os.path.exists(draft_folder):
            raise HTTPException(status_code=404, detail="Generated draft files not found.")
            
        # 2. Di chuyển các tài liệu nháp sang lưu trữ bài tập chính thức
        os.makedirs(official_folder, exist_ok=True)
        for filename in os.listdir(draft_folder):
            shutil.move(os.path.join(draft_folder, filename), os.path.join(official_folder, filename))
            
        statement_path = os.path.join(official_folder, "statement.md")
        theory_path = os.path.join(official_folder, "theory.md")
        tutorial_path = os.path.join(official_folder, "tutorial.md")
        solution_path = os.path.join(official_folder, "solution.py")
        coding_path = os.path.join(official_folder, "coding.py")
        checker_path = os.path.join(official_folder, "checker.py")
        temp_input_dir = os.path.join(official_folder, "inputs")
        temp_output_dir = os.path.join(official_folder, "outputs")
        input_zip_path = os.path.join(official_folder, "input.zip")
        output_zip_path = os.path.join(official_folder, "output.zip")
        
        # 3. Ghi nhận bài tập vào bảng problems
        cursor.execute("""
            INSERT INTO problems (
                name, source, statement_path, theory_path, tutorial_path, 
                solution_path, coding_path, checker_path, author_id, 
                is_public, request_status, input_folder_path, output_folder_path,
                input_zip_path, output_zip_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'NONE', ?, ?, ?, ?)
        """, (
            name,
            f"Roadmap: {roadmap_id}",
            statement_path,
            theory_path,
            tutorial_path,
            solution_path,
            coding_path,
            checker_path,
            user_id,
            temp_input_dir,
            temp_output_dir,
            input_zip_path,
            output_zip_path
        ))
        problem_id = cursor.lastrowid
        
        # 4. Liên kết ID bài tập chính thức và đổi status thành 'saved'
        cursor.execute("""
            UPDATE roadmap_problems
            SET problem_id = ?, status = 'saved'
            WHERE id = ?
        """, (problem_id, step_id))
        
        db.commit()
        
        # Dọn dẹp thư mục nháp tạm thời
        if os.path.exists(draft_folder):
            shutil.rmtree(draft_folder)
            
        return {
            "status": "success",
            "message": "Step converted and saved to real problems table successfully.",
            "data": {
                "problem_id": problem_id,
                "step_id": step_id,
                "status": "saved"
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save to real problem: {str(e)}")
    
@app.get("/api/problems/draft_sessions/{session_id}")
def get_draft_session_detail(session_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, roadmap_name, repository_url, user_id, problems_json, num_test_cases, status, created_at, updated_at
            FROM draft_problem_sessions
            WHERE id = ?
        """, (session_id,))
        session = cursor.fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            proposed_problems = [ProposedProblem(**item) for item in json.loads(session["problems_json"])]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Invalid stored draft JSON: {str(e)}")

        return {
            "status": "success",
            "data": {
                "session_id": session["id"],
                "roadmap_name": session["roadmap_name"],
                "repository_url": session["repository_url"],
                "user_id": session["user_id"],
                "status": session["status"],
                "num_test_cases": session["num_test_cases"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "proposed_problems": proposed_problems
            }
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
