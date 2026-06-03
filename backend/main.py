import datetime
import secrets
import sqlite3
from google import genai
import jwt
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header, Response, Cookie
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
from typing import List, Optional, Any, Dict
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
from prompts.prompt import get_problems_from_repo_prompt, feedback_prompt, create_detailedly_prompt, validate_problem_from_repo_prompt, validate_create_detailedly_response_prompt
from dotenv import load_dotenv
from json_repair import repair_json
import tempfile
import subprocess
import sys

os.makedirs("database", exist_ok=True)
os.makedirs("storage/problems", exist_ok=True)
os.makedirs("storage/avatars", exist_ok=True)

app = FastAPI()
app.mount("/avatars", StaticFiles(directory="storage/avatars"), name="avatars")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:21080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_path = "storage/"
deepwiki_url = "http://localhost:21082"
load_dotenv()
EMAIL_FORMAT_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client()

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
    conn = sqlite3.connect("database/database.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()

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
    approve: bool
    
class DetailedProblemMaterials(BaseModel):
    statement: str = Field(..., description="Markdown description of the problem and requirements")
    theory: str = Field(..., description="Markdown DL/ML theory basis")
    tutorial: str = Field(..., description="Markdown step-by-step guidance on how to solve the problem")
    solution: str = Field(..., description="Complete solution code")
    coding: str = Field(..., description="Python template or code file content")
    checker: str = Field(..., description="Checker code for validating submissions")
    test_inputs: str = Field(..., description="A JSON array string containing 3 testcase input objects")

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
    num_test_cases: Optional[int] = 3
    
class ProposedProblem(BaseModel):
    title: str = Field(..., description="Problem title")
    description: str = Field(..., description="Detailed description")
    target_module: str = Field(..., description="Path to target module")

class ProblemsFromRepoDataResponse(BaseModel):
    session_id: int
    repository_url: str
    roadmap_name: str
    proposed_problems: List[ProposedProblem]

class ProblemsFromRepoResponse(BaseModel):
    status: str
    message: str
    data: ProblemsFromRepoDataResponse

class MakeContributorRequest(BaseModel):
    user_id: int
    secret_key: str

class ProblemUpdatePayload(BaseModel):
    user_id: int
    name: Optional[str] = None
    source: Optional[str] = None
    statement_markdown: Optional[str] = None
    theory_markdown: Optional[str] = None
    tutorial_markdown: Optional[str] = None
    solution_markdown: Optional[str] = None
    coding_markdown: Optional[str] = None
    checker_markdown: Optional[str] = None

class ProblemDeletePayload(BaseModel):
    user_id: int

class BlogCreate(BaseModel):
    title: str
    content: str
    author_id: int

class CommentCreate(BaseModel):
    content: str
    user_id: int
    problem_id: Optional[int] = None
    blog_id: Optional[int] = None
    parent_id: Optional[int] = None

class CommentUpdate(BaseModel):
    content: str
    user_id: int

class CommentDeletePayload(BaseModel):
    user_id: int

class VoteRequest(BaseModel):
    user_id: int
    blog_id: Optional[int] = None
    comment_id: Optional[int] = None
    vote_type: int # 1: Upvote, -1: Downvote

class TicketCreate(BaseModel):
    user_id: int
    title: str
    description: str

class TicketReplyCreate(BaseModel):
    user_id: int
    message: str

class TicketStatusUpdate(BaseModel):
    user_id: int
    status: str # 'open' or 'resolved'

class AdminUserUpdatePayload(BaseModel):
    admin_id: Optional[int] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None





































def check_wiki_cache(owner: str, repo: str, repo_type: str = "github", language: str = "en"):  
    params = {  
        "owner": owner,  
        "repo": repo,  
        "repo_type": repo_type,  
        "language": language  
    }  
    response = requests.get(f"{deepwiki_url}/api/wiki_cache", params=params)  
    return response.json() if response.status_code == 200 else None  

def ask_question(repo_url: str, question: str, provider: str = "google"):  
    payload = {  
        "repo_url": repo_url,  
        "messages": [{"role": "user", "content": question}],  
        "provider": provider  
    }  
    response = requests.post(f"{deepwiki_url}/chat/completions/stream", json=payload, stream=True)  
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
    required_fields = {"statement", "theory", "tutorial", "solution", "coding"}
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        raise ValueError(f"AI response JSON is incomplete. Missing fields: {', '.join(sorted(missing_fields))}.")
    return data

def parse_and_repair_json(raw_text: str) -> dict:
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
    if not raw_text:
        raise ValueError("Empty AI response")

    text = raw_text.strip()

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if m:
        return m.group(1).strip()

    return text


def compare_approx(v1: Any, v2: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(v1, list) and isinstance(v2, list):
        if len(v1) != len(v2):
            return False
        return all(compare_approx(x, y, tolerance) for x, y in zip(v1, v2))
    elif isinstance(v1, dict) and isinstance(v2, dict):
        if set(v1.keys()) != set(v2.keys()):
            return False
        return all(compare_approx(v1[k], v2[k], tolerance) for k in v1)
    elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return abs(v1 - v2) / max(1.0, abs(v2)) <= tolerance
    return v1 == v2

def extract_python_code(markdown_content: Optional[str]) -> str:
    if not markdown_content:
        return ""
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", markdown_content, re.DOTALL)
    return match.group(1).strip() if match else markdown_content.strip()

def validate_email_format(email: str) -> Optional[str]:
    if not EMAIL_FORMAT_PATTERN.match(email):
        return "Email must be a valid format."
    return None

def fix_problem_from_repo_response(raw_ai_response: str) -> SyntaxWarning:
    prompt = validate_problem_from_repo_prompt.format(last_ai_response=raw_ai_response)
    ai_response_fixed = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        config={
            "system_instruction": "You are a strict JSON Validator and Senior Deep Learning Engineer. Your task is to ingest a raw, potentially malformed or technically inaccurate JSON string representing a roadmap of programming tasks, repair it, and output a standardized, production-ready JSON array. Follow the objectives and constraints outlined in the prompt meticulously."
        },
        contents=f"text:{prompt}"
    ).text

    return ai_response_fixed

def fix_create_detailed_response(raw_ai_response: str) -> str:
    prompt = validate_create_detailedly_response_prompt.format(raw_ai_response=raw_ai_response)
    ai_response_fixed = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        config={
            "system_instruction": "You are a strict JSON Validator and Senior Deep Learning Engineer. Your task is to ingest a raw, potentially malformed or technically inaccurate JSON string representing detailed problem materials, repair it, and output a standardized, production-ready JSON object. Follow the objectives and constraints outlined in the prompt meticulously."
        },
        contents=f"text:{prompt}"
    ).text

    return ai_response_fixed







































# ================= API ENDPOINTS =================

@app.post("/api/auth/register")
def register(user: UserRegister, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    username = user.username.strip()
    display_name = user.display_name.strip()
    email = user.email.strip()
    password = user.password

    missing_fields = []
    if not username: missing_fields.append("username")
    if not display_name: missing_fields.append("display_name")
    if not email: missing_fields.append("email")
    if not password or not password.strip(): missing_fields.append("password")

    if missing_fields:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing_fields)}")

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
def login(credentials: UserLogin, response: Response, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (credentials.username,))
    user = cursor.fetchone()
    
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if user["status"] != "active":
        raise HTTPException(status_code=403, detail="Account is disabled")
    
    access_token = create_access_token(
        data={
            "sub": user["username"], 
            "role": user["role"],
            "user_id": user["id"]
        },
        expires_delta=datetime.timedelta(minutes=30)
    )
    
    refresh_token = secrets.token_hex(32)
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
    
    cursor.execute(
        "INSERT INTO refresh_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user["id"], refresh_token, expires_at.strftime('%Y-%m-%d %H:%M:%S'))
    )
    db.commit()
    
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        max_age=7 * 24 * 3600,
        samesite="lax",
        secure=False
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user["id"],
        "user_role": user["role"],
        "avatar_url": user["avatar_url"]
    }
    
@app.post("/api/auth/refresh")
def refresh(response: Response, refresh_token: Optional[str] = Cookie(None), db: sqlite3.Connection = Depends(get_db)):
    if not refresh_token:
         raise HTTPException(status_code=401, detail="Session expired. Please re-login.")
         
    cursor = db.cursor()
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute(
        "SELECT * FROM refresh_tokens WHERE token = ? AND expires_at > ? AND is_revoked = 0", 
        (refresh_token, now_utc)
    )
    token_record = cursor.fetchone()
    if not token_record:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    cursor.execute("SELECT id, username, role, status FROM users WHERE id = ?", (token_record["user_id"],))
    user = cursor.fetchone()
    if not user or user["status"] != "active":
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    
    new_access_token = create_access_token(
        data={
            "sub": user["username"], 
            "role": user["role"],
            "user_id": user["id"]
        },
        expires_delta=datetime.timedelta(minutes=30)
    )
    return {"access_token": new_access_token, "token_type": "bearer"}

@app.post("/api/auth/logout")
def logout(response: Response, refresh_token: Optional[str] = Cookie(None), db: sqlite3.Connection = Depends(get_db)):
    if refresh_token:
        cursor = db.cursor()
        cursor.execute("DELETE FROM refresh_tokens WHERE token = ?", (refresh_token,))
        db.commit()
    response.delete_cookie("refresh_token")
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
    cursor.execute("SELECT role FROM users WHERE id = ?", (author_id,))
    user_row = cursor.fetchone()
    if not user_row or user_row["role"] not in ("admin", "contributor"):
        raise HTTPException(status_code=403, detail="Unauthorized. Only admins or contributors can create problems.")

    try:
        name_slug = normalize_problem_name(name)

        cursor.execute("SELECT id FROM problems WHERE name = ?", (name_slug,))

        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="The problem name already exists.")
        print("cc")

        problem_folder = initialize_problem_storage(storage_path, name_slug)
        statement_path = os.path.join(problem_folder, "statement.md")
        theory_path = os.path.join(problem_folder, "theory.md") if theory_markdown else None
        tutorial_path = os.path.join(problem_folder, "tutorial.md") if tutorial_markdown else None
        solution_path = os.path.join(problem_folder, "solution.py") if solution_markdown else None
        coding_path = os.path.join(problem_folder, "coding.py") if coding_markdown else None
        checker_path = os.path.join(problem_folder, "checker.py") if checker_markdown else None

        with open(statement_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(statement_markdown.replace("\r\n", "\n"))
        if theory_path:
            with open(theory_path, "w", encoding="utf-8", newline="\n") as f: f.write(theory_markdown.replace("\r\n", "\n"))
        if tutorial_path:
            with open(tutorial_path, "w", encoding="utf-8", newline="\n") as f: f.write(tutorial_markdown.replace("\r\n", "\n"))
        if solution_path:
            with open(solution_path, "w", encoding="utf-8", newline="\n") as f: f.write(solution_markdown.replace("\r\n", "\n"))
        if coding_path:
            with open(coding_path, "w", encoding="utf-8", newline="\n") as f: f.write(coding_markdown.replace("\r\n", "\n"))
        if checker_path:
            with open(checker_path, "w", encoding="utf-8", newline="\n") as f: f.write(checker_markdown.replace("\r\n", "\n"))

        input_folder_path = None
        output_folder_path = None
        if input_zip:
            input_folder_path = save_and_unzip_file(problem_folder, input_zip, "inputs")
            if not validate_folder_structure(input_folder_path):
                raise HTTPException(status_code=400, detail="Invalid input folder structure")
        if output_zip:
            output_folder_path = save_and_unzip_file(problem_folder, output_zip, "outputs")
            if not validate_folder_structure(output_folder_path):
                raise HTTPException(status_code=400, detail="Invalid output folder structure")

        cursor.execute("""
            INSERT INTO problems (
                name, source, statement_path, theory_path, tutorial_path, 
                solution_path, coding_path, checker_path, author_id, 
                is_public, request_status, input_folder_path, output_folder_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'NONE', ?, ?)
        """, (name, source, statement_path, theory_path, tutorial_path, solution_path, coding_path, checker_path, author_id, input_folder_path, output_folder_path))

        problem_id = cursor.lastrowid
        db.commit()
        return {
            "status": "success",
            "success": True,
            "message": "Create problem manually successfully",
            "data": {
                "id": problem_id,
                "problem_id": problem_id
            },
            "id": problem_id,
            "problem_id": problem_id
        }
    except Exception as e:
        db.rollback()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Usecase: Fetch and filter problems
@app.get("/api/problems/filter")
def filter_problems(user_id: int = None, filter_mode: str = "public", db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        if user_id:
            select_clause = """
                p.id, p.name, p.source, p.author_id, p.is_public, p.request_status, p.created_at,
                (SELECT MAX(s.score) FROM submissions s WHERE s.problem_id = p.id AND s.user_id = ?) as best_score,
                (SELECT s.status FROM submissions s WHERE s.problem_id = p.id AND s.user_id = ? ORDER BY s.score DESC, s.created_at DESC LIMIT 1) as best_status
            """
        else:
            select_clause = """
                p.id, p.name, p.source, p.author_id, p.is_public, p.request_status, p.created_at,
                NULL as best_score, NULL as best_status
            """

        if filter_mode == "public":
            if user_id:
                cursor.execute(f"SELECT {select_clause} FROM problems p WHERE p.is_public = 1 AND p.request_status IN ('APPROVED', 'NONE') ORDER BY p.created_at DESC", (user_id, user_id))
            else:
                cursor.execute(f"SELECT {select_clause} FROM problems p WHERE p.is_public = 1 AND p.request_status IN ('APPROVED', 'NONE') ORDER BY p.created_at DESC")
        elif filter_mode == "private":
            if not user_id:
                raise HTTPException(status_code=400, detail="user_id required for private filter")
            cursor.execute(f"SELECT {select_clause} FROM problems p WHERE p.author_id = ? ORDER BY p.created_at DESC", (user_id, user_id, user_id))
        elif filter_mode == "all":
            if not user_id:
                raise HTTPException(status_code=400, detail="user_id required for all filter")
            cursor.execute(f"SELECT {select_clause} FROM problems p WHERE p.is_public = 1 OR p.author_id = ? ORDER BY p.created_at DESC", (user_id, user_id, user_id))
        else:
            raise HTTPException(status_code=400, detail="Invalid filter_mode.")
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

# Usecase: Get problem content to display in livecoding page
@app.get("/api/problems/{problem_id}/content")
def get_problem_content(problem_id: int, user_id: Optional[int] = None, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, name, source, statement_path, theory_path, tutorial_path, solution_path, coding_path, checker_path
            FROM problems
            WHERE id = ?
        """, (problem_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Problem not found")

        # Xác định vai trò của người dùng yêu cầu
        role = "user"
        if user_id:
            cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            if user:
                role = user["role"]

        def read_file(path_value: Optional[str]) -> str:
            if not path_value or not os.path.exists(path_value):
                return ""
            with open(path_value, "r", encoding="utf-8") as f:
                content = f.read()
            return content.replace("\r\n", "\n")

        data = dict(row)
        statement_markdown = read_file(data["statement_path"])
        theory_markdown = read_file(data["theory_path"])
        tutorial_markdown = read_file(data["tutorial_path"])
        solution_markdown = read_file(data["solution_path"])
        coding_markdown = read_file(data["coding_path"])
        checker_markdown = read_file(data["checker_path"]) if role in ("admin", "contributor") else ""

        if role not in ("admin", "contributor"):
            has_solved = False
            if user_id:
                cursor.execute("""
                    SELECT 1 FROM submissions 
                    WHERE problem_id = ? AND user_id = ? AND (score = 100 OR status = 'accepted') 
                    LIMIT 1
                """, (problem_id, user_id))
                if cursor.fetchone():
                    has_solved = True
            
            if not has_solved:
                solution_markdown = "## Restricted Access\nYou must solve this problem with an Accepted status (100 pts) to view the sample solution."

        return {
            "status": "success",
            "data": {
                "id": data["id"],
                "name": data["name"],
                "source": data["source"],
                "statement_markdown": statement_markdown,
                "theory_markdown": theory_markdown,
                "tutorial_markdown": tutorial_markdown,
                "solution_markdown": solution_markdown,
                "coding_markdown": coding_markdown,
                "checker_markdown": checker_markdown,
            }
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

# Usecase: Run code on the first test case (for quick feedback in livecoding)
@app.post("/api/problems/{problem_id}/run")
async def run_problem_code(
    problem_id: int,
    payload: RunRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    submitted_code = payload.submitted_code
    cursor = db.cursor()
    
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

    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])
    if not input_files:
        raise HTTPException(status_code=400, detail="No input JSON files found in testcases")
        
    first_input_file = input_files[0]
    input_file_path = os.path.join(input_folder, first_input_file)
    abs_input_file_path = os.path.abspath(input_file_path)
    
    output_file_path = os.path.join(output_folder, first_input_file)
    if not os.path.exists(output_file_path):
        output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.json')])
        if not output_files:
            raise HTTPException(status_code=400, detail="No output JSON files found in testcases")
        output_file_path = os.path.join(output_folder, output_files[0])
        
    abs_output_file_path = os.path.abspath(output_file_path)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_out:
        temp_output_path = tmp_out.name
    abs_temp_output_path = os.path.abspath(temp_output_path)

    escaped_input_path = abs_input_file_path.replace('\\', '\\\\')
    escaped_output_path = abs_temp_output_path.replace('\\', '\\\\')
    
    submitted_code = submitted_code.replace("input.json", escaped_input_path)
    submitted_code = submitted_code.replace("output.json", escaped_output_path)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as temp_file:
        temp_file.write(submitted_code)
        temp_file_path = temp_file.name

    start_time = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=30.0
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
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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
        if os.path.exists(abs_temp_output_path):
            os.remove(abs_temp_output_path)

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

    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])
    if not input_files:
        raise HTTPException(status_code=400, detail="No input JSON files found for evaluation")

    total_tc = len(input_files)
    passed_tc = 0
    db_status = "accepted"
    results: List[Dict[str, Any]] = []

    for idx, in_file in enumerate(input_files):
        abs_in_path = os.path.abspath(os.path.join(input_folder, in_file))
        abs_out_path = os.path.abspath(os.path.join(output_folder, in_file))
        if not os.path.exists(abs_out_path):
            output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.json')])
            if len(output_files) > idx:
                abs_out_path = os.path.abspath(os.path.join(output_folder, output_files[idx]))
            else:
                results.append({
                    "testcase": in_file,
                    "status": "System Error",
                    "user_output": "Missing expected output on server"
                })
                db_status = "wrong_answer" if db_status == "accepted" else db_status
                continue

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_out:
            temp_output_path = tmp_out.name
        abs_temp_output_path = os.path.abspath(temp_output_path)

        escaped_input_path = abs_in_path.replace('\\', '\\\\')
        escaped_output_path = abs_temp_output_path.replace('\\', '\\\\')
        
        tc_submitted_code = submitted_code.replace("input.json", escaped_input_path)
        tc_submitted_code = tc_submitted_code.replace("output.json", escaped_output_path)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(tc_submitted_code)
            temp_file_path = temp_file.name

        tc_status = "Wrong Answer"
        user_output_display = ""
        
        try:
            process = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            if process.returncode != 0:
                tc_status = "Runtime Error"
                user_output_display = process.stderr.strip()[:150]
                if db_status in ("accepted", "wrong_answer"):
                    db_status = "runtime_error"
            else:
                try:
                    if not os.path.exists(abs_temp_output_path) or os.path.getsize(abs_temp_output_path) == 0:
                        raise ValueError("No output generated")
                        
                    with open(abs_temp_output_path, "r", encoding="utf-8") as f:
                        user_output_json = json.load(f)
                    
                    user_output_display = str(user_output_json)[:150]
                    
                    with open(abs_out_path, "r", encoding="utf-8") as f:
                        expected_output_json = json.load(f)

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
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(abs_temp_output_path):
                os.remove(abs_temp_output_path)

        results.append({
            "testcase": in_file,
            "status": tc_status,
            "user_output": user_output_display
        })

    score = int((passed_tc / total_tc) * 100) if total_tc > 0 else 0

    status_mapping = {
        "accepted": "Accepted",
        "wrong_answer": "Wrong Answer",
        "runtime_error": "Runtime Error",
        "time_limit_exceeded": "Time Limit Exceeded"
    }
    friendly_status = status_mapping.get(db_status, "Wrong Answer")

    test_results_json_str = json.dumps(results)
    
    try:
        cursor.execute("""
            INSERT INTO submissions (user_id, problem_id, submitted_code, status, score, test_results)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, problem_id, submitted_code, db_status, score, test_results_json_str))
        db.commit()
        submission_id = cursor.lastrowid
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database write error: {str(e)}")

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

def require_admin_user(cursor: sqlite3.Cursor, authorization: Optional[str]) -> dict:
    identity = get_authenticated_identity(authorization)
    authenticated_user_id = identity.get("user_id")
    authenticated_username = identity.get("username", "")

    if authenticated_user_id is not None:
        cursor.execute(
            "SELECT id, username, role, status FROM users WHERE id = ?",
            (authenticated_user_id,)
        )
    else:
        cursor.execute(
            "SELECT id, username, role, status FROM users WHERE LOWER(username) = LOWER(?)",
            (authenticated_username,)
        )

    admin = cursor.fetchone()
    if not admin or admin["role"] != "admin" or admin["status"] != "active":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return dict(admin)

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

        if avatar is not None:
            allowed_types = {"image/jpeg", "image/png", "image/webp"}
            if avatar.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail="Avatar must be JPG, PNG, or WEBP image")
            file_bytes = avatar.file.read()
            max_size = 5 * 1024 * 1024
            if len(file_bytes) > max_size:
                raise HTTPException(status_code=400, detail="Avatar file size must be <= 5MB")
            ext_map = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
            ext = ext_map.get(avatar.content_type, "jpg")
            filename = f"{uuid.uuid4()}.{ext}"
            avatar_path = os.path.join("storage", "avatars", filename)
            with open(avatar_path, "wb") as out_file:
                out_file.write(file_bytes)
            avatar_url = f"/avatars/{filename}"
            update_fields.append("avatar_url = ?")
            params.append(avatar_url) 
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

@app.post("/api/users/make-contributor")
def make_user_contributor(payload: MakeContributorRequest, db: sqlite3.Connection = Depends(get_db)):
    contributor_secret = os.getenv("CONTRIBUTOR_SECRET_KEY", "SUPER_SECRET_CONTRIBUTOR_2026")
    if payload.secret_key != contributor_secret:
        raise HTTPException(status_code=400, detail="Invalid contributor secret key. Access denied.")
        
    cursor = db.cursor()
    try:
        cursor.execute("UPDATE users SET role = 'contributor' WHERE id = ?", (payload.user_id,))
        db.commit()
        return {
            "status": "success",
            "message": "Congratulations! You have successfully upgraded to Contributor. Please re-login to apply changes."
        }
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database execution error: {str(e)}")
    
# Usecase : Add admin role to a user (for admin management)
@app.post("/api/users/make-admin")
def make_user_admin(payload: MakeAdminRequest, db: sqlite3.Connection = Depends(get_db)):
    admin_secret = os.getenv("ADMIN_SECRET_KEY", "SUPER_SECRET_ADMIN_2026")
    if payload.secret_key != admin_secret:
        raise HTTPException(status_code=400, detail="Invalid admin secret key. Access denied.")
        
    cursor = db.cursor()
    try:
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
        try:
            json_string = clean_json_response(ai_raw_response)
            parsed_problems = [ProposedProblem(**item) for item in json.loads(json_string)]
        except Exception as e:
            print(f"[Error] Initial AI response parsing failed: {str(e)}")
            num_tries = 3
            for _ in range(num_tries):
                try:
                    print(f"[Info] Attempting to fix AI response format... (Attempt {_ + 1}/{num_tries})")
                    ai_raw_response_fixed = fix_problem_from_repo_response(ai_raw_response)
                    json_string = clean_json_response(ai_raw_response_fixed)
                    parsed_problems = [ProposedProblem(**item) for item in json.loads(json_string)]
                    print(f"[Info] Successfully fixed AI response on attempt {_ + 1}")
                    break
                except json.JSONDecodeError:
                    print(f"[Error] AI response still has invalid JSON format. Attempting to fix again... (Attempt {_ + 1}/{num_tries})")
                    if (_ == num_tries - 1):
                        raise HTTPException(status_code=502, detail="AI generated an invalid format after multiple attempts.")
                    
        cursor = db.cursor()
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
        raise HTTPException(status_code=502, detail="AI generated an invalid format.")
    except ValidationError as e:
        raise HTTPException(status_code=502, detail="AI response did not match the schema structure.")
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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

# Usecase : Admin xem danh sách toàn bộ người dùng hệ thống (Sử dụng cho UsersTab)
@app.get("/api/admin/users")
def list_managed_users(
    admin_id: Optional[int] = None,
    search: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    try:
        is_authenticated = False
        if authorization:
            try:
                require_admin_user(cursor, authorization)
                is_authenticated = True
            except HTTPException:
                pass

        if not is_authenticated:
            if admin_id is not None:
                cursor.execute("SELECT role, status FROM users WHERE id = ?", (admin_id,))
                user = cursor.fetchone()
                if not user or user["role"] != "admin" or user["status"] != "active":
                    raise HTTPException(status_code=403, detail="Admin privileges required")
            else:
                raise HTTPException(status_code=401, detail="Authentication required")

        params = []
        where_clause = ""
        if search and search.strip():
            term = f"%{search.strip().lower()}%"
            where_clause = """
                WHERE LOWER(username) LIKE ?
                   OR LOWER(display_name) LIKE ?
                   OR LOWER(email) LIKE ?
                   OR LOWER(role) LIKE ?
                   OR LOWER(status) LIKE ?
            """
            params = [term, term, term, term, term]

        cursor.execute(f"""
            SELECT id, username, display_name, email, role, status, avatar_url, created_at
            FROM users
            {where_clause}
            ORDER BY created_at DESC, id DESC
        """, params)
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.post("/api/problems/draft_sessions/feedback")
def update_draft_session_with_feedback(payload: FeedbackRequest, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("SELECT repository_url, problems_json FROM draft_problem_sessions WHERE id = ?", (payload.session_id,))
        session = cursor.fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        current_problems = session["problems_json"]
        repo_url = session["repository_url"]
        
        question_prompt = feedback_prompt.format(
            repo_url=repo_url,
            current_problems=current_problems,
            payload=payload
        )
        
        ai_response = ask_question(repo_url, question_prompt)
        json_string = clean_json_response(ai_response)
        parsed_problems = [ProposedProblem(**item) for item in json.loads(json_string)]
        
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
            # Tạo roadmap nháp
            cursor.execute("""
                INSERT INTO roadmaps (user_id, name, repository_url, level, num_test_cases, status)
                VALUES (?, ?, ?, ?, ?, 'draft')
            """, (user_id, payload.roadmap_title, repo_url, "medium", num_test_cases))
            roadmap_id = cursor.lastrowid
            
            created_problems = []
            for index, prob in enumerate(problems_list, start=1):
                name = prob["title"].strip()
                description = prob.get("description", "").strip()
                target_module = prob.get("target_module", "").strip()
                
                # CHỈ thêm vào roadmap_problems với problem_id = NULL. 
                # Không gọi INSERT INTO problems ở đây để tránh hiển thị sớm trên tab Problems.
                cursor.execute("""
                    INSERT INTO roadmap_problems (roadmap_id, problem_id, name, order_index, status, description, target_module)
                    VALUES (?, NULL, ?, ?, 'pending', ?, ?)
                """, (roadmap_id, name, index, description, target_module))
                step_id = cursor.lastrowid
                
                created_problems.append({
                    "step_id": step_id,
                    "name": name,
                    "order_index": index,
                    "description": description,
                    "target_module": target_module
                })
                
            cursor.execute("UPDATE draft_problem_sessions SET status = 'finalized' WHERE id = ?", (payload.session_id,))
            db.commit()
            
            return {
                "status": "success", 
                "message": "Roadmap created successfully. Steps initialized as private drafts.",
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
        data["problems"] = steps
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
    
    draft_folder = f"{storage_path}draft_problems/{step_id}"
    
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
    with open("debug_ai_response.txt", "w", encoding="utf-8") as debug_file:
        debug_file.write(ai_raw_response)
    
    ai_raw_response = fix_create_detailed_response(ai_raw_response)
    with open("debug_ai_response_fixed.txt", "w", encoding="utf-8") as debug_file:
        debug_file.write(ai_raw_response)
    try:
        raw_json_data = parse_and_repair_json(ai_raw_response)
        raw_json_data = validate_and_ensure_complete_materials(raw_json_data)
        materials = DetailedProblemMaterials(**raw_json_data)
        
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

        for i in range(1, num_test_cases + 1):
            inp = test_inputs_list[i - 1]
            inp_file_path = os.path.join(temp_input_dir, f"input_{i}.json")
            out_file_path = os.path.join(temp_output_dir, f"output_{i}.json")
            
            with open(inp_file_path, "w", encoding="utf-8") as f_in:
                json.dump(inp, f_in)
                
            temp_input_json = os.path.join(draft_folder, "input.json")
            temp_output_json = os.path.join(draft_folder, "output.json")
            
            with open(temp_input_json, "w", encoding="utf-8") as f_temp:
                json.dump(inp, f_temp)
                
            try:
                process = subprocess.run(
                    [sys.executable, "solution.py"],
                    cwd=draft_folder,
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                
                if process.returncode == 0 and os.path.exists(temp_output_json):
                    shutil.copy2(temp_output_json, out_file_path)
                else:
                    print(f"[ERROR] Generator failed for testcase {i}")
                    with open(out_file_path, "w", encoding="utf-8") as f_out:
                        json.dump({"error": "failed"}, f_out)
            except Exception as e:
                with open(out_file_path, "w", encoding="utf-8") as f_out:
                    json.dump({"error": str(e)}, f_out)
            finally:
                if os.path.exists(temp_input_json):
                    os.remove(temp_input_json)
                if os.path.exists(temp_output_json):
                    os.remove(temp_output_json)

        input_zip_path = os.path.join(draft_folder, "input.zip")
        output_zip_path = os.path.join(draft_folder, "output.zip")

        with zipfile.ZipFile(input_zip_path, 'w') as zip_in:
            for file in sorted(os.listdir(temp_input_dir)):
                zip_in.write(os.path.join(temp_input_dir, file), arcname=file)

        with zipfile.ZipFile(output_zip_path, 'w') as zip_out:
            for file in sorted(os.listdir(temp_output_dir)):
                zip_out.write(os.path.join(temp_output_dir, file), arcname=file)

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
        
        cursor.execute("SELECT id FROM problems WHERE name = ?", (name_slug,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="The problem name already exists.")
            
        draft_folder = f"{storage_path}draft_problems/{step_id}"
        official_folder = f"{storage_path}problems/{name_slug}"
        
        if not os.path.exists(draft_folder):
            raise HTTPException(status_code=404, detail="Generated draft files not found.")
            
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
        
        cursor.execute("""
            UPDATE roadmap_problems
            SET problem_id = ?, status = 'saved'
            WHERE id = ?
        """, (problem_id, step_id))
        
        db.commit()
        
        if os.path.exists(draft_folder):
            shutil.rmtree(draft_folder)
            
        return {
            "status": "success",
            "message": "Step converted and saved successfully.",
            "data": {
                "problem_id": problem_id,
                "step_id": step_id,
                "status": "saved"
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")
    
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

# Usecase: Lấy danh sách testcase của bài tập
@app.get("/api/problems/{problem_id}/testcases")
def get_problem_testcases(problem_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT input_folder_path, output_folder_path FROM problems WHERE id = ?", (problem_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    input_folder = row["input_folder_path"]
    output_folder = row["output_folder_path"]
    
    if not input_folder or not os.path.exists(input_folder):
        return {"status": "success", "data": []}
        
    try:
        input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])
        output_files = []
        if output_folder and os.path.exists(output_folder):
            output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.json')])
            
        testcases = []
        for idx, f in enumerate(input_files, start=1):
            inp_path = os.path.join(input_folder, f)
            
            with open(inp_path, "r", encoding="utf-8") as file_in:
                inp_content = file_in.read()
            
            out_content = ""
            out_path = None
            if output_folder and os.path.exists(output_folder):
                guess_names = [f"output_{idx}.json", f"input_{idx}.json", f]
                for g_name in guess_names:
                    p = os.path.join(output_folder, g_name)
                    if os.path.exists(p):
                        out_path = p
                        break
                if not out_path and len(output_files) >= idx:
                    out_path = os.path.join(output_folder, output_files[idx - 1])
                    
            if out_path and os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as file_out:
                    out_content = file_out.read()
                    
            testcases.append({
                "id": idx,
                "input": inp_content,
                "output": out_content
            })
        return {"status": "success", "data": testcases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Usecase : Sửa bài tập cho contribution và admin
@app.put("/api/problems/{problem_id}")
async def update_problem(
    problem_id: int,
    user_id: int = Form(...),
    name: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    statement_markdown: Optional[str] = Form(None),
    theory_markdown: Optional[str] = Form(None),
    tutorial_markdown: Optional[str] = Form(None),
    solution_markdown: Optional[str] = Form(None),
    coding_markdown: Optional[str] = Form(None),
    checker_markdown: Optional[str] = Form(None),
    testcases: Optional[str] = Form(None),
    input_zip: Optional[UploadFile] = File(None),   
    output_zip: Optional[UploadFile] = File(None), 
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user or user["role"] not in ("admin", "contributor"):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    cursor.execute("SELECT author_id, request_status, statement_path, theory_path, tutorial_path, solution_path, coding_path, checker_path, input_folder_path, output_folder_path FROM problems WHERE id = ?", (problem_id,))
    prob = cursor.fetchone()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    if user["role"] == "contributor":
        if prob["author_id"] != user_id:
            raise HTTPException(status_code=403, detail="You can only edit your own problems")
    
    update_fields = []
    params = []
    
    if name:
        update_fields.append("name = ?")
        params.append(name)
    if source is not None:
        update_fields.append("source = ?")
        params.append(source)
        
    problem_folder = os.path.dirname(prob["statement_path"])
        
    if statement_markdown is not None and prob["statement_path"]:
        with open(prob["statement_path"], "w", encoding="utf-8", newline="\n") as f:
            f.write(statement_markdown.replace("\r\n", "\n"))
    if theory_markdown is not None and prob["theory_path"]:
        with open(prob["theory_path"], "w", encoding="utf-8", newline="\n") as f:
            f.write(theory_markdown.replace("\r\n", "\n"))
    if tutorial_markdown is not None and prob["tutorial_path"]:
        with open(prob["tutorial_path"], "w", encoding="utf-8", newline="\n") as f:
            f.write(tutorial_markdown.replace("\r\n", "\n"))
    if solution_markdown is not None and prob["solution_path"]:
        with open(prob["solution_path"], "w", encoding="utf-8", newline="\n") as f:
            f.write(solution_markdown.replace("\r\n", "\n"))
    if coding_markdown is not None and prob["coding_path"]:
        with open(prob["coding_path"], "w", encoding="utf-8", newline="\n") as f:
            f.write(coding_markdown.replace("\r\n", "\n"))
    if checker_markdown is not None and prob["checker_path"]:
        with open(prob["checker_path"], "w", encoding="utf-8", newline="\n") as f:
            f.write(checker_markdown.replace("\r\n", "\n"))
            
    if input_zip:
        if prob["input_folder_path"] and os.path.exists(prob["input_folder_path"]):
            try:
                shutil.rmtree(prob["input_folder_path"])
            except Exception:
                pass
        input_folder_path = save_and_unzip_file(problem_folder, input_zip, "inputs")
        if not validate_folder_structure(input_folder_path):
            raise HTTPException(status_code=400, detail="Invalid input folder structure")
        update_fields.append("input_folder_path = ?")
        params.append(input_folder_path)

    if output_zip:
        if prob["output_folder_path"] and os.path.exists(prob["output_folder_path"]):
            try:
                shutil.rmtree(prob["output_folder_path"])
            except Exception:
                pass
        output_folder_path = save_and_unzip_file(problem_folder, output_zip, "outputs")
        if not validate_folder_structure(output_folder_path):
            raise HTTPException(status_code=400, detail="Invalid output folder structure")
        update_fields.append("output_folder_path = ?")
        params.append(output_folder_path)

    if testcases is not None and not input_zip and not output_zip:
        input_folder = prob["input_folder_path"]
        output_folder = prob["output_folder_path"]
        
        if not input_folder:
            input_folder = os.path.join(problem_folder, "inputs")
            update_fields.append("input_folder_path = ?")
            params.append(input_folder)
            
        if not output_folder:
            output_folder = os.path.join(problem_folder, "outputs")
            update_fields.append("output_folder_path = ?")
            params.append(output_folder)
            
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        # Clear existing
        for f in os.listdir(input_folder):
            try:
                os.remove(os.path.join(input_folder, f))
            except Exception:
                pass
        for f in os.listdir(output_folder):
            try:
                os.remove(os.path.join(output_folder, f))
            except Exception:
                pass
                
        try:
            tc_list = json.loads(testcases)
            for idx, tc in enumerate(tc_list, start=1):
                inp_filename = f"input_{idx}.json"
                out_filename = f"output_{idx}.json"
                inp_path = os.path.join(input_folder, inp_filename)
                out_path = os.path.join(output_folder, out_filename)
                
                with open(inp_path, "w", encoding="utf-8") as f_in:
                    parsed_inp = json.loads(tc["input"]) if isinstance(tc["input"], str) else tc["input"]
                    json.dump(parsed_inp, f_in, indent=2)
                    
                with open(out_path, "w", encoding="utf-8") as f_out:
                    parsed_out = json.loads(tc["output"]) if isinstance(tc["output"], str) else tc["output"]
                    json.dump(parsed_out, f_out, indent=2)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process testcases: {str(e)}")
            
    if update_fields:
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        cursor.execute(f"UPDATE problems SET {', '.join(update_fields)} WHERE id = ?", params + [problem_id])
        db.commit()
        
    return {"status": "success", "message": "Problem updated successfully"}

# Usecase : Xóa bài tập cho contribution và admin
@app.delete("/api/problems/{problem_id}")
def delete_problem(problem_id: int, payload: ProblemDeletePayload, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT role FROM users WHERE id = ?", (payload.user_id,))
    user = cursor.fetchone()
    if not user or user["role"] not in ("admin", "contributor"):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    cursor.execute("SELECT author_id, request_status, statement_path, theory_path, tutorial_path, solution_path, coding_path, checker_path FROM problems WHERE id = ?", (problem_id,))
    prob = cursor.fetchone()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    if user["role"] == "contributor":
        if prob["author_id"] != payload.user_id:
            raise HTTPException(status_code=403, detail="You can only delete your own problems")
            
    for key in ["statement_path", "theory_path", "tutorial_path", "solution_path", "coding_path", "checker_path"]:
        p = prob[key]
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
                
    cursor.execute("DELETE FROM problems WHERE id = ?", (problem_id,))
    db.commit()
    return {"status": "success", "message": "Problem deleted successfully"}

@app.get("/api/blogs")
def get_blogs(user_id: Optional[int] = None, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT b.id, b.title, b.content, b.author_id, b.created_at, b.updated_at,
                   u.display_name AS author_name, u.avatar_url AS author_avatar,
                   COALESCE((SELECT SUM(vote_type) FROM votes WHERE blog_id = b.id), 0) AS score,
                   COALESCE((SELECT vote_type FROM votes WHERE blog_id = b.id AND user_id = ?), 0) AS user_vote
            FROM blogs b
            JOIN users u ON b.author_id = u.id
            ORDER BY b.created_at DESC
        """, (user_id,))
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blogs/{blog_id}")
def get_blog_detail(blog_id: int, user_id: Optional[int] = None, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT b.id, b.title, b.content, b.author_id, b.created_at, b.updated_at,
                   u.display_name AS author_name, u.avatar_url AS author_avatar,
                   COALESCE((SELECT SUM(vote_type) FROM votes WHERE blog_id = b.id), 0) AS score,
                   COALESCE((SELECT vote_type FROM votes WHERE blog_id = b.id AND user_id = ?), 0) AS user_vote
            FROM blogs b
            JOIN users u ON b.author_id = u.id
            WHERE b.id = ?
        """, (user_id, blog_id))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Blog post not found")
        return {"status": "success", "data": dict(row)}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/blogs")
def create_blog(payload: BlogCreate, db: sqlite3.Connection = Depends(get_db)):
    if not payload.title.strip() or not payload.content.strip():
        raise HTTPException(status_code=400, detail="Title and content cannot be empty")
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO blogs (title, content, author_id) VALUES (?, ?, ?)",
            (payload.title.strip(), payload.content.strip(), payload.author_id)
        )
        db.commit()
        return {"status": "success", "message": "Blog created successfully", "id": cursor.lastrowid}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# 5.1 & 5.2 & 5.3: Quản lý Thảo luận / Bình luận (Comments)
@app.get("/api/comments")
def get_comments(
    problem_id: Optional[int] = None,
    blog_id: Optional[int] = None,
    user_id: Optional[int] = None,
    db: sqlite3.Connection = Depends(get_db)
):
    if not problem_id and not blog_id:
        raise HTTPException(status_code=400, detail="Either problem_id or blog_id must be provided")
    cursor = db.cursor()
    try:
        if problem_id:
            cursor.execute("""
                SELECT c.id, c.content, c.user_id, c.problem_id, c.blog_id, c.parent_id, c.created_at, c.updated_at,
                       u.display_name AS user_name, u.role AS user_role, u.avatar_url AS user_avatar,
                       COALESCE((SELECT SUM(vote_type) FROM votes WHERE comment_id = c.id), 0) AS score,
                       COALESCE((SELECT vote_type FROM votes WHERE comment_id = c.id AND user_id = ?), 0) AS user_vote
                FROM comments c
                JOIN users u ON c.user_id = u.id
                WHERE c.problem_id = ?
                ORDER BY c.created_at ASC
            """, (user_id, problem_id))
        else:
            cursor.execute("""
                SELECT c.id, c.content, c.user_id, c.problem_id, c.blog_id, c.parent_id, c.created_at, c.updated_at,
                       u.display_name AS user_name, u.role AS user_role, u.avatar_url AS user_avatar,
                       COALESCE((SELECT SUM(vote_type) FROM votes WHERE comment_id = c.id), 0) AS score,
                       COALESCE((SELECT vote_type FROM votes WHERE comment_id = c.id AND user_id = ?), 0) AS user_vote
                FROM comments c
                JOIN users u ON c.user_id = u.id
                WHERE c.blog_id = ?
                ORDER BY c.created_at ASC
            """, (user_id, blog_id))
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/comments")
def create_comment(payload: CommentCreate, db: sqlite3.Connection = Depends(get_db)):
    if not payload.content or not payload.content.strip():
        raise HTTPException(status_code=400, detail="Bình luận không được phép bỏ trống.")
    
    cursor = db.cursor()
    try:
        cursor.execute("""
            INSERT INTO comments (content, user_id, problem_id, blog_id, parent_id)
            VALUES (?, ?, ?, ?, ?)
        """, (payload.content.strip(), payload.user_id, payload.problem_id, payload.blog_id, payload.parent_id))
        db.commit()
        return {"status": "success", "message": "Bình luận thành công", "id": cursor.lastrowid}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/comments/{comment_id}")
def update_comment(comment_id: int, payload: CommentUpdate, db: sqlite3.Connection = Depends(get_db)):
    if not payload.content or not payload.content.strip():
        raise HTTPException(status_code=400, detail="Bình luận không được phép bỏ trống.")
    cursor = db.cursor()
    try:
        cursor.execute("SELECT user_id FROM comments WHERE id = ?", (comment_id,))
        comment = cursor.fetchone()
        if not comment:
            raise HTTPException(status_code=404, detail="Bình luận không tồn tại")
        
        # Chỉ người viết mới được sửa bình luận của họ
        if comment["user_id"] != payload.user_id:
            raise HTTPException(status_code=403, detail="Bạn không thể sửa bình luận của người khác")
            
        cursor.execute("UPDATE comments SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (payload.content.strip(), comment_id))
        db.commit()
        return {"status": "success", "message": "Cập nhật bình luận thành công"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/comments/{comment_id}/delete")
def delete_comment(comment_id: int, payload: CommentDeletePayload, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("SELECT user_id FROM comments WHERE id = ?", (comment_id,))
        comment = cursor.fetchone()
        if not comment:
            raise HTTPException(status_code=404, detail="Bình luận không tồn tại")
            
        cursor.execute("SELECT role FROM users WHERE id = ?", (payload.user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
            
        is_owner = comment["user_id"] == payload.user_id
        is_admin = user["role"] == "admin"
        
        if not is_owner and not is_admin:
            raise HTTPException(status_code=403, detail="Bạn không có quyền xóa bình luận này")
            
        cursor.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
        db.commit()
        return {"status": "success", "message": "Xóa bình luận thành công"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# 5.4: Hệ thống Upvote / Downvote (Bình chọn)
@app.post("/api/votes")
def handle_vote(payload: VoteRequest, db: sqlite3.Connection = Depends(get_db)):
    if not payload.blog_id and not payload.comment_id:
        raise HTTPException(status_code=400, detail="Phải cung cấp blog_id hoặc comment_id")
        
    cursor = db.cursor()
    try:
        if payload.blog_id:
            cursor.execute("SELECT vote_type FROM votes WHERE user_id = ? AND blog_id = ?", (payload.user_id, payload.blog_id))
            row = cursor.fetchone()
            if row:
                if row["vote_type"] == payload.vote_type:
                    # Bấm lại nút cũ -> Hủy đánh giá
                    cursor.execute("DELETE FROM votes WHERE user_id = ? AND blog_id = ?", (payload.user_id, payload.blog_id))
                else:
                    # Bấm nút ngược lại -> Cập nhật trạng thái
                    cursor.execute("UPDATE votes SET vote_type = ? WHERE user_id = ? AND blog_id = ?", (payload.vote_type, payload.user_id, payload.blog_id))
            else:
                # Tạo đánh giá mới
                cursor.execute("INSERT INTO votes (user_id, blog_id, vote_type) VALUES (?, ?, ?)", (payload.user_id, payload.blog_id, payload.vote_type))
        else:
            cursor.execute("SELECT vote_type FROM votes WHERE user_id = ? AND comment_id = ?", (payload.user_id, payload.comment_id))
            row = cursor.fetchone()
            if row:
                if row["vote_type"] == payload.vote_type:
                    # Bấm lại nút cũ -> Hủy đánh giá
                    cursor.execute("DELETE FROM votes WHERE user_id = ? AND comment_id = ?", (payload.user_id, payload.comment_id))
                else:
                    # Bấm nút ngược lại -> Cập nhật
                    cursor.execute("UPDATE votes SET vote_type = ? WHERE user_id = ? AND comment_id = ?", (payload.vote_type, payload.user_id, payload.comment_id))
            else:
                cursor.execute("INSERT INTO votes (user_id, comment_id, vote_type) VALUES (?, ?, ?)", (payload.user_id, payload.comment_id, payload.vote_type))
                
        db.commit()
        return {"status": "success", "message": "Xử lý bình chọn thành công"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# 5.5: Hệ thống Quản lý Ticket Hỗ trợ
@app.post("/api/tickets")
def create_ticket(payload: TicketCreate, db: sqlite3.Connection = Depends(get_db)):
    if not payload.title.strip() or not payload.description.strip():
        raise HTTPException(status_code=400, detail="Vui lòng điền tiêu đề và mô tả sự cố")
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO tickets (user_id, title, description, status) VALUES (?, ?, ?, 'open')",
            (payload.user_id, payload.title.strip(), payload.description.strip())
        )
        db.commit()
        return {"status": "success", "message": "Tạo Ticket hỗ trợ thành công", "id": cursor.lastrowid}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tickets")
def list_tickets(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
            
        # Admin xem tất cả, User/Contributor xem ticket cá nhân
        if user["role"] == "admin":
            cursor.execute("""
                SELECT t.id, t.user_id, t.title, t.description, t.status, t.created_at, t.updated_at,
                       u.display_name AS creator_name
                FROM tickets t
                JOIN users u ON t.user_id = u.id
                ORDER BY t.created_at DESC
            """)
        else:
            cursor.execute("""
                SELECT t.id, t.user_id, t.title, t.description, t.status, t.created_at, t.updated_at,
                       u.display_name AS creator_name
                FROM tickets t
                JOIN users u ON t.user_id = u.id
                WHERE t.user_id = ?
                ORDER BY t.created_at DESC
            """, (user_id,))
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tickets/{ticket_id}")
def get_ticket_detail(ticket_id: int, user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
            
        cursor.execute("""
            SELECT t.id, t.user_id, t.title, t.description, t.status, t.created_at, t.updated_at,
                   u.display_name AS creator_name
            FROM tickets t
            JOIN users u ON t.user_id = u.id
            WHERE t.id = ?
        """, (ticket_id,))
        ticket = cursor.fetchone()
        if not ticket:
            raise HTTPException(status_code=404, detail="Không tìm thấy Ticket hỗ trợ")
            
        if user["role"] != "admin" and ticket["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Bạn không có quyền xem Ticket này")
            
        cursor.execute("""
            SELECT r.id, r.user_id, r.message, r.created_at,
                   u.display_name AS replier_name, u.role AS replier_role, u.avatar_url AS replier_avatar
            FROM ticket_replies r
            JOIN users u ON r.user_id = u.id
            WHERE r.ticket_id = ?
            ORDER BY r.created_at ASC
        """, (ticket_id,))
        replies = [dict(row) for row in cursor.fetchall()]
        
        return {
            "status": "success",
            "data": {
                **dict(ticket),
                "replies": replies
            }
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tickets/{ticket_id}/replies")
def create_ticket_reply(ticket_id: int, payload: TicketReplyCreate, db: sqlite3.Connection = Depends(get_db)):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Nội dung phản hồi không thể để trống")
    cursor = db.cursor()
    try:
        cursor.execute("SELECT role FROM users WHERE id = ?", (payload.user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
            
        cursor.execute("SELECT user_id FROM tickets WHERE id = ?", (ticket_id,))
        ticket = cursor.fetchone()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket không tồn tại")
            
        if user["role"] != "admin" and ticket["user_id"] != payload.user_id:
            raise HTTPException(status_code=403, detail="Không có quyền phản hồi Ticket này")
            
        cursor.execute("""
            INSERT INTO ticket_replies (ticket_id, user_id, message)
            VALUES (?, ?, ?)
        """, (ticket_id, payload.user_id, payload.message.strip()))
        
        cursor.execute("UPDATE tickets SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (ticket_id,))
        db.commit()
        return {"status": "success", "message": "Gửi phản hồi thành công"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tickets/{ticket_id}/status")
def update_ticket_status(ticket_id: int, payload: TicketStatusUpdate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("SELECT role FROM users WHERE id = ?", (payload.user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
            
        cursor.execute("SELECT user_id FROM tickets WHERE id = ?", (ticket_id,))
        ticket = cursor.fetchone()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket không tồn tại")
            
        if user["role"] != "admin" and ticket["user_id"] != payload.user_id:
            raise HTTPException(status_code=403, detail="Không có quyền cập nhật trạng thái")
            
        cursor.execute("UPDATE tickets SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (payload.status, ticket_id))
        db.commit()
        return {"status": "success", "message": f"Cập nhật trạng thái thành công"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    

@app.put("/api/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    payload: AdminUserUpdatePayload,
    admin_id: Optional[int] = None,
    authorization: Optional[str] = Header(None),
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    
    # Xác định id của admin thực hiện yêu cầu
    effective_admin_id = payload.admin_id or admin_id
    
    is_authenticated = False
    if authorization:
        try:
            require_admin_user(cursor, authorization)
            is_authenticated = True
        except HTTPException:
            pass

    if not is_authenticated:
        if effective_admin_id is not None:
            cursor.execute("SELECT role, status FROM users WHERE id = ?", (effective_admin_id,))
            user = cursor.fetchone()
            if not user or user["role"] != "admin" or user["status"] != "active":
                raise HTTPException(status_code=403, detail="Admin privileges required")
        else:
            raise HTTPException(status_code=401, detail="Authentication required")

    # Kiểm tra sự tồn tại của người dùng mục tiêu
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()
    if not target_user:
        raise HTTPException(status_code=404, detail="Target user not found")

    # Tiến hành cập nhật thông tin
    update_fields = []
    params = []

    if payload.username is not None:
        username = payload.username.strip()
        if username:
            cursor.execute("SELECT 1 FROM users WHERE id != ? AND LOWER(username) = LOWER(?)", (user_id, username))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Username already exists")
            update_fields.append("username = ?")
            params.append(username)

    if payload.display_name is not None:
        display_name = payload.display_name.strip()
        if display_name:
            update_fields.append("display_name = ?")
            params.append(display_name)

    if payload.email is not None:
        email = payload.email.strip()
        if email:
            email_error = validate_email_format(email)
            if email_error:
                raise HTTPException(status_code=400, detail=email_error)
            cursor.execute("SELECT 1 FROM users WHERE id != ? AND LOWER(email) = LOWER(?)", (user_id, email))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Email already exists")
            update_fields.append("email = ?")
            params.append(email)

    if payload.role is not None:
        role = payload.role.strip()
        if role in ("admin", "contributor", "user"):
            update_fields.append("role = ?")
            params.append(role)

    if payload.status is not None:
        status = payload.status.strip()
        if status in ("active", "disabled", "inactive"):
            update_fields.append("status = ?")
            params.append(status)

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    update_fields.append("updated_at = CURRENT_TIMESTAMP")
    
    try:
        cursor.execute(f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?", params + [user_id])
        db.commit()
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {"status": "success", "message": "User updated successfully by admin"}