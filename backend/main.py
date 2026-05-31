import datetime
import secrets
import sqlite3
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ValidationError
from backend.auth import hash_password, verify_password, create_access_token
from typing import List, Optional
import re
import os
import requests  
import json
from prompts.prompt import get_problems_from_repo_prompt, feedback_prompt, create_detailedly_prompt, correction_prompt
from dotenv import load_dotenv
from json_repair import repair_json


os.makedirs("database", exist_ok=True)
os.makedirs("storage/problems", exist_ok=True)

app = FastAPI()

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
    author_id: int            
    
class ProblemsFromRepo(BaseModel):
    roadmap_name: str
    repository_url: str
    level: str
    user_id: int
    user_note: Optional[str] = ""
    framework: Optional[str] = ""
    
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

class SubmissionCreate(BaseModel):
    problem_id: int
    submitted_code: str

class UserProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    email: Optional[EmailStr] = None


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
    



def unescape_json_strings(data):
    """
    Recursively unescape JSON string values (convert \\n to actual newlines, etc.)
    """
    if isinstance(data, dict):
        return {k: unescape_json_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [unescape_json_strings(item) for item in data]
    elif isinstance(data, str):
        # Decode escape sequences: \\n -> \n, \\t -> \t, etc.
        return data.encode().decode('unicode_escape')
    else:
        return data


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
    
        
    pattern = r'"(statement|theory|tutorial|solution|coding)"\s*:\s*"((?:[^"\\]|\\.)*)"'
    matches = re.findall(pattern, raw_text)
    extracted_fields = {}
    for key, value in matches:
        decoded_value = value.encode('utf-8').decode('unicode_escape')
        extracted_fields[key] = decoded_value

    return extracted_fields


# ================= API ENDPOINTS =================

@app.post("/api/auth/register")
def register(user: UserRegister, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (user.username, user.email))
    if cursor.fetchone(): 
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    hashed_password = hash_password(user.password)
    
    try:
        cursor.execute(
            """INSERT INTO users (username, password_hash, display_name, email) VALUES (?, ?, ?, ?)""", 
            (user.username, hashed_password, user.display_name, user.email)
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    return {"message": "User registered successfully"}
        
@app.post("/api/auth/login")
def login(credentials: UserLogin, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (credentials.username,))
    user = cursor.fetchone()
    
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=datetime.timedelta(minutes=30)
    )
    
    refresh_token = secrets.token_hex(32)
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
    
    cursor.execute(
        "INSERT INTO refresh_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user["id"], refresh_token, expires_at.strftime('%Y-%m-%d %H:%M:%S'))
    )
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
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
    
    cursor.execute("SELECT username, role FROM users WHERE id = ?", (token_record["user_id"],))
    user = cursor.fetchone()
    
    new_access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
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

@app.post("/api/problems/create/manual")
def create_problem_manual(payload: ManualProblemCreate, db: sqlite3.Connection = Depends(get_db)):
    name = payload.name.strip()
    if not name: 
        raise HTTPException(status_code=400, detail="Problem name cannot be empty")
    if not payload.statement_markdown.strip():
        raise HTTPException(status_code=400, detail="Problem statement cannot be empty")

    cursor = db.cursor()
    
    try:
        # Check if the name already exists
        cursor.execute("SELECT id FROM problems WHERE name = ?", (name,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=400, 
                detail="The problem name already exists. Please choose a different name."
            )
        
        name_slug = normalize_problem_name(name)
        problem_folder = f"{storage_path}problems/{name_slug}"
        os.makedirs(problem_folder, exist_ok=True)
        
        statement_path = os.path.join(problem_folder, "statement.md")
        theory_path = os.path.join(problem_folder, "theory.md") if payload.theory_markdown else None
        tutorial_path = os.path.join(problem_folder, "tutorial.md") if payload.tutorial_markdown else None
        solution_path = os.path.join(problem_folder, "solution.md") if payload.solution_markdown else None
        coding_path = os.path.join(problem_folder, "coding.md") if payload.coding_markdown else None

        with open(statement_path, "w", encoding="utf-8") as f:
            f.write(payload.statement_markdown)

        if theory_path:
            with open(theory_path, "w", encoding="utf-8") as f:
                f.write(payload.theory_markdown)
                
        if tutorial_path:
            with open(tutorial_path, "w", encoding="utf-8") as f:
                f.write(payload.tutorial_markdown)
        
        if solution_path:
            with open(solution_path, "w", encoding="utf-8") as f:
                f.write(payload.solution_markdown)
        
        if coding_path:
            with open(coding_path, "w", encoding="utf-8") as f:
                f.write(payload.coding_markdown)

        # Insert with new field names: name, author_id, request_status
        cursor.execute("""
            INSERT INTO problems (name, source, statement_path, theory_path, tutorial_path, solution_path, coding_path, author_id, is_public, request_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 'NONE')
        """, (
            name,
            payload.source,
            statement_path,
            theory_path,
            tutorial_path,
            solution_path,
            coding_path,
            payload.author_id
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
                "author_id": payload.author_id,
                "is_public": 1,
                "request_status": "NONE"
            }
        }
        
    except sqlite3.Error as e:
            db.rollback()
            print(f"\n[DEBUG] SQLite Error: {e}\n")
            raise HTTPException(
                status_code=500, 
                detail=f"Database query error: {str(e)}"
            )
            
    except IOError as e:
        db.rollback()
        print(f"\n[DEBUG] File IO Error: {e}\n")
        raise HTTPException(
            status_code=500, 
            detail=f"File write error: {str(e)}"
        )

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
        return {
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
    except sqlite3.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database query error: {str(e)}"
        )


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

        # Ask AI to generate proposed problems based on the repository and user notes
        ai_raw_response = ask_question(payload.repository_url, question_prompt)
        print(f"\n[DEBUG] AI Response from DeepWiki/Groq:\n{ai_raw_response}\n")

        # Clean the AI response to extract the JSON string, then parse it into Python objects
        json_string = clean_json_response(ai_raw_response)
        raw_json_data = json.loads(json_string)

        parsed_problems = [ProposedProblem(**item) for item in raw_json_data]

        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO draft_problem_sessions (roadmap_name, repository_url, user_id, problems_json, status)
            VALUES (?, ?, ?, ?, 'draft')
        """, (payload.roadmap_name, payload.repository_url, payload.user_id, json_string))
        
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
    cursor.execute("""
        SELECT id, roadmap_name, repository_url, status, created_at 
        FROM draft_problem_sessions 
        WHERE user_id = ? AND status = 'draft'
        ORDER BY id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    return {"status": "success", "data": [dict(row) for row in rows]}

@app.get("/api/problems/draft_sessions/{session_id}")
def get_draft_session_detail(session_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("""
        SELECT id, roadmap_name, repository_url, user_id, problems_json, status, created_at, updated_at
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
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "proposed_problems": proposed_problems
        }
    }

@app.post("/api/problems/draft_sessions/feedback")
def update_draft_session_with_feedback(payload: FeedbackRequest, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    
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
    
    try:
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
    
    # 1. Read draft session information
    cursor.execute("""
        SELECT repository_url, user_id, problems_json, status 
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
    problems_list = json.loads(session["problems_json"])
    
    try:
        # 2. Create official research roadmap
        cursor.execute("""
            INSERT INTO roadmaps (user_id, name, repository_url, level, status)
            VALUES (?, ?, ?, ?, 'draft')
        """, (user_id, payload.roadmap_title, repo_url, "medium"))
        roadmap_id = cursor.lastrowid
        
        # 3. Create child problems as private drafts
        created_problems = []
        for index, prob in enumerate(problems_list, start=1):
            name = prob["title"].strip()
            name_slug = normalize_problem_name(name)
            
            problem_folder = f"{storage_path}problems/{name_slug}"
            os.makedirs(problem_folder, exist_ok=True)
            
            # Create temporary statement.md
            statement_path = os.path.join(problem_folder, "statement.md")
            with open(statement_path, "w", encoding="utf-8") as f:
                f.write(f"# {name}\n\n*Problem details are being initialized by AI. Please click 'Create Detailedly' to complete.*")
            
            # INSERT into problems table as private draft
            cursor.execute("""
                INSERT INTO problems (name, source, statement_path, author_id, is_public, request_status)
                VALUES (?, ?, ?, ?, 0, 'NONE')
            """, (name, f"Roadmap: {payload.roadmap_title}", statement_path, user_id))
            problem_id = cursor.lastrowid
            
            # Link problem to roadmap
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
            
        # 4. Update draft session status to finalized
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

@app.get("/api/roadmaps/{roadmap_id}")
def get_roadmap_detail(roadmap_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, name, repository_url, level, user_note, framework, status, user_id, created_at
            FROM roadmaps
            WHERE id = ?
        """, (roadmap_id,))
        roadmap = cursor.fetchone()
        if not roadmap:
            raise HTTPException(status_code=404, detail="Roadmap not found")

        cursor.execute("""
            SELECT p.id AS problem_id, p.name, rp.order_index, p.statement_path, p.request_status, p.is_public,
                   p.theory_path, p.tutorial_path, p.solution_path, p.coding_path
            FROM roadmap_problems rp
            JOIN problems p ON p.id = rp.problem_id
            WHERE rp.roadmap_id = ?
            ORDER BY rp.order_index ASC
        """, (roadmap_id,))
        problems_raw = [dict(row) for row in cursor.fetchall()]
        
        # Add has_materials flag to indicate if problem has all detailed materials
        problems = []
        for problem in problems_raw:
            problem_dict = dict(problem)
            # Check if all material files are present (not null)
            has_all_materials = all([
                problem_dict.get("statement_path"),
                problem_dict.get("theory_path"),
                problem_dict.get("tutorial_path"),
                problem_dict.get("solution_path"),
                problem_dict.get("coding_path")
            ])
            problem_dict["has_materials"] = has_all_materials
            problems.append(problem_dict)

        data = dict(roadmap)
        data["problems"] = problems
        return {"status": "success", "data": data}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.get("/api/roadmaps")
def list_roadmaps(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT r.id, r.name, r.repository_url, r.level, r.status, r.created_at,
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

@app.post("/api/problems/{problem_id}/create_detailedly")
def create_problem_detailedly(problem_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    
    # 1. Query problem name, repository_url, and roadmap name from related tables
    cursor.execute("""
        SELECT p.name, r.repository_url, r.name AS roadmap_name 
        FROM problems p
        JOIN roadmap_problems rp ON p.id = rp.problem_id
        JOIN roadmaps r ON rp.roadmap_id = r.id
        WHERE p.id = ?
    """, (problem_id,))
    record = cursor.fetchone()
    
    if not record:
        raise HTTPException(
            status_code=404, 
            detail="Problem not found or it is not associated with any active Roadmap."
        )
        
    name = record["name"]
    repo_url = record["repository_url"]
    roadmap_name = record["roadmap_name"]
    name_slug = normalize_problem_name(name)
    problem_folder = f"{storage_path}problems/{name_slug}"
    
    # 2. Create prompt requesting DeepWiki to write detailed educational materials
    try:
        question_prompt = create_detailedly_prompt.format(
            title=name,
            repository_url=repo_url,
            roadmap_title=roadmap_name
        )
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Prompt formatting error: Missing key {str(e)}")

    # 3. Send query to DeepWiki
    ai_raw_response = ask_question(repo_url, question_prompt)
    with open("debug_ai_response.txt", "w", encoding="utf-8") as f:
        f.write(ai_raw_response)
    
    
    try:
        # 4. Use self-healing JSON parser and repair with Groq AI
        raw_json_data = parse_and_repair_json(ai_raw_response)
        print(f"\n[DEBUG] Parsed JSON data keys: {list(raw_json_data.keys())}")
        
        # 4b. Validate that all required fields are present and non-empty
        raw_json_data = validate_and_ensure_complete_materials(raw_json_data)
        
        # 5. Validate with Pydantic
        materials = DetailedProblemMaterials(**raw_json_data)
        
        # 6. Write detailed content to files on disk
        statement_path = os.path.join(problem_folder, "statement.md")
        theory_path = os.path.join(problem_folder, "theory.md")
        tutorial_path = os.path.join(problem_folder, "tutorial.md")
        solution_path = os.path.join(problem_folder, "solution.md")
        coding_path = os.path.join(problem_folder, "coding.md")
        
        os.makedirs(problem_folder, exist_ok=True)
        
        with open(statement_path, "w", encoding="utf-8") as f:
            f.write(materials.statement)
            
        with open(theory_path, "w", encoding="utf-8") as f:
            f.write(materials.theory)
            
        with open(tutorial_path, "w", encoding="utf-8") as f:
            f.write(materials.tutorial)
        
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(materials.solution)
            
        with open(coding_path, "w", encoding="utf-8") as f:
            f.write(materials.coding)
            
        # 7. Update file paths in database
        cursor.execute("""
            UPDATE problems 
            SET theory_path = ?, tutorial_path = ?, solution_path = ?, coding_path = ? 
            WHERE id = ?
        """, (theory_path, tutorial_path, solution_path, coding_path, problem_id))
        
        db.commit()
        
        return {
            "status": "success",
            "message": f"Successfully generated educational materials for problem '{name}'",
            "data": {
                "problem_id": problem_id,
                "name": name,
                "statement_path": statement_path,
                "theory_path": theory_path,
                "tutorial_path": tutorial_path,
                "solution_path": solution_path,
                "coding_path": coding_path
            }
        }
        
    except ValueError as e:
        # This catches JSON parsing/validation errors
        print(f"\n[ERROR] ValueError during JSON parsing: {str(e)}")
        raise HTTPException(
            status_code=502, 
            detail=f"Failed to parse and repair AI response: {str(e)}"
        )
    except ValidationError as e:
        # This catches Pydantic schema validation errors
        error_details = e.errors()
        print(f"\n[ERROR] Pydantic Validation Errors:")
        for error in error_details:
            print(f"  - Field '{error['loc'][0]}': {error['msg']}")
        
        # More informative error message
        missing_fields = [str(err['loc'][0]) for err in error_details if err['type'] == 'missing']
        if missing_fields:
            detail = f"AI response is missing required fields: {', '.join(missing_fields)}. " \
                     f"The AI response might be truncated or incomplete. Please try again."
        else:
            detail = f"AI response has validation errors. Check the server logs for details."
        
        raise HTTPException(
            status_code=502,
            detail=detail
        )
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Database error during material generation: {str(e)}"
        )
    except IOError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"File system write error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error: {str(e)}"
        )

# ================= USER PROFILE ENDPOINTS =================

@app.get("/api/users/{user_id}")
def get_user_profile(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, username, display_name, email, avatar_url, role, created_at
            FROM users
            WHERE id = ?
        """, (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Count problems solved by this user
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM submissions
            WHERE user_id = ? AND status = 'accepted'
        """, (user_id,))
        solved = cursor.fetchone()["count"]
        
        data = dict(user)
        data["problems_solved"] = solved
        return {"status": "success", "data": data}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.put("/api/users/{user_id}")
def update_user_profile(user_id: int, payload: UserProfileUpdate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="User not found")
        
        # Build dynamic update query
        updates = []
        params = []
        if payload.display_name:
            updates.append("display_name = ?")
            params.append(payload.display_name)
        if payload.email:
            updates.append("email = ?")
            params.append(payload.email)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(user_id)
        
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        db.commit()
        
        return {"status": "success", "message": "User profile updated successfully"}
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ================= SUBMISSIONS / LIVE CODING ENDPOINTS =================

@app.post("/api/submissions")
def create_submission(payload: SubmissionCreate, user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # Verify problem exists
        cursor.execute("SELECT id FROM problems WHERE id = ?", (payload.problem_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Problem not found")
        
        # Create submission record
        cursor.execute("""
            INSERT INTO submissions (user_id, problem_id, submitted_code, status)
            VALUES (?, ?, ?, 'pending')
        """, (user_id, payload.problem_id, payload.submitted_code))
        db.commit()
        submission_id = cursor.lastrowid
        
        return {
            "status": "success",
            "message": "Submission created successfully",
            "data": {
                "submission_id": submission_id,
                "problem_id": payload.problem_id,
                "status": "pending"
            }
        }
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/submissions/{submission_id}")
def get_submission(submission_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT id, user_id, problem_id, submitted_code, status, score, test_results, created_at
            FROM submissions
            WHERE id = ?
        """, (submission_id,))
        submission = cursor.fetchone()
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        return {"status": "success", "data": dict(submission)}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.get("/api/users/{user_id}/submissions")
def list_user_submissions(user_id: int, problem_id: int = None, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        if problem_id:
            cursor.execute("""
                SELECT id, problem_id, status, score, created_at
                FROM submissions
                WHERE user_id = ? AND problem_id = ?
                ORDER BY created_at DESC
            """, (user_id, problem_id))
        else:
            cursor.execute("""
                SELECT id, problem_id, status, score, created_at
                FROM submissions
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
        
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

# ================= PROBLEM FILTERING ENDPOINTS =================

@app.get("/api/problems/filter")
def filter_problems(user_id: int = None, filter_mode: str = "public", db: sqlite3.Connection = Depends(get_db)):
    """
    Filter problems by mode:
    - public: Only public problems (is_public = 1)
    - private: Only user's private problems (is_public = 0, author_id = user_id)
    - all: All problems (requires user_id)
    """
    cursor = db.cursor()
    try:
        if filter_mode == "public":
            cursor.execute("""
                SELECT id, name, source, author_id, is_public, request_status, created_at
                FROM problems
                WHERE is_public = 1 AND request_status IN ('APPROVED', 'NONE')
                ORDER BY created_at DESC
            """)
        elif filter_mode == "private":
            if not user_id:
                raise HTTPException(status_code=400, detail="user_id required for private filter")
            cursor.execute("""
                SELECT id, name, source, author_id, is_public, request_status, created_at
                FROM problems
                WHERE is_public = 0 AND author_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
        elif filter_mode == "all":
            if not user_id:
                raise HTTPException(status_code=400, detail="user_id required for all filter")
            cursor.execute("""
                SELECT id, name, source, author_id, is_public, request_status, created_at
                FROM problems
                WHERE is_public = 1 OR author_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
        else:
            raise HTTPException(status_code=400, detail="Invalid filter_mode. Use: public, private, or all")
        
        return {"status": "success", "data": [dict(row) for row in cursor.fetchall()]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

# ================= PROBLEM APPROVAL WORKFLOW ENDPOINTS =================

@app.post("/api/problems/{problem_id}/request-approval")
def request_problem_approval(problem_id: int, db: sqlite3.Connection = Depends(get_db)):
    """User requests admin approval to make problem public"""
    cursor = db.cursor()
    try:
        # Check if problem exists and belongs to user
        cursor.execute("""
            SELECT id, request_status, author_id
            FROM problems
            WHERE id = ?
        """, (problem_id,))
        problem = cursor.fetchone()
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        if problem["request_status"] == "PENDING":
            raise HTTPException(status_code=400, detail="Problem approval already requested")
        
        cursor.execute("""
            UPDATE problems
            SET request_status = 'PENDING'
            WHERE id = ?
        """, (problem_id,))
        db.commit()
        
        return {
            "status": "success",
            "message": "Approval request submitted successfully"
        }
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/problems/{problem_id}/approve")
def approve_problem(problem_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Admin approves a problem for public release"""
    cursor = db.cursor()
    try:
        # Check if problem exists
        cursor.execute("SELECT id, request_status FROM problems WHERE id = ?", (problem_id,))
        problem = cursor.fetchone()
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        cursor.execute("""
            UPDATE problems
            SET request_status = 'APPROVED', is_public = 1
            WHERE id = ?
        """, (problem_id,))
        db.commit()
        
        return {
            "status": "success",
            "message": "Problem approved and made public successfully"
        }
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/problems/{problem_id}/reject")
def reject_problem(problem_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Admin rejects a problem approval request"""
    cursor = db.cursor()
    try:
        cursor.execute("""
            UPDATE problems
            SET request_status = 'REJECTED'
            WHERE id = ?
        """, (problem_id,))
        db.commit()
        
        return {
            "status": "success",
            "message": "Problem approval request rejected"
        }
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        