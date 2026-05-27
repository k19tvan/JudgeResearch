import sqlite3
import hashlib
import os

db_name = "database.db"
def get_db_connection():
    con = sqlite3.connect(f"database/{db_name}")
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = get_db_connection()
    cursor = con.cursor()
    
    # =============== Users Table ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        display_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        avatar_url TEXT,
        role TEXT NOT NULL DEFAULT 'user',
        status TEXT NOT NULL DEFAULT 'active',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # =============== Refresh Tokens Table ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS refresh_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token TEXT UNIQUE NOT NULL,
        expires_at DATETIME NOT NULL,
        is_revoked INTEGER NOT NULL DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # =============== Problems Table ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS problems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        author_id INTEGER,
        statement_path TEXT NOT NULL,
        theory_path TEXT,
        tutorial_path TEXT,
        solution_path TEXT,
        coding_path TEXT,
        source TEXT,
        input_zip_url TEXT,
        output_zip_url TEXT,
        is_public INTEGER NOT NULL DEFAULT 0,
        request_status TEXT NOT NULL DEFAULT 'NONE',  -- NONE, PENDING, APPROVED, REJECTED
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (author_id) REFERENCES users(id) ON DELETE SET NULL
    );
    """)
    
    # =============== Submissions Table ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        problem_id INTEGER NOT NULL,
        submitted_code TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',  -- pending, accepted, wrong_answer, runtime_error, time_limit_exceeded
        score INTEGER DEFAULT 0,
        test_results TEXT,  -- JSON with detailed test results
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
    );
    """)
    
    # =============== Draft Problem Sessions (Intermediate table for iterative refinement) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS draft_problem_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_name TEXT,
        repository_url TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        problems_json TEXT NOT NULL,         
        status TEXT NOT NULL DEFAULT 'draft', -- 'draft' (đang sửa) hoặc 'finalized' (đã duyệt)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # =============== Roadmaps Table ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS roadmaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        repository_url TEXT NOT NULL,
        level TEXT NOT NULL,  -- easy, medium, hard
        user_note TEXT,
        framework TEXT,  -- pytorch, tensorflow, etc.
        status TEXT NOT NULL DEFAULT 'draft',  -- draft, in_progress, completed
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== Roadmap Problems Table ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS roadmap_problems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_id INTEGER NOT NULL,
        problem_id INTEGER,  -- NULL during initial phase before problems are created
        name TEXT NOT NULL,
        description TEXT,
        order_index INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',  -- pending, in_progress, completed
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE SET NULL
    );
    """)
    
    con.commit()
    con.close()
    
    print("Database initialized successfully.")
    
if __name__ == "__main__":
    init_db()
