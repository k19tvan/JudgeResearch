import sqlite3
import os

db_dir = "database"
db_name = "database.db"
db_path = os.path.join(db_dir, db_name)

# Đảm bảo thư mục lưu trữ cơ sở dữ liệu tồn tại
os.makedirs(db_dir, exist_ok=True)

def get_db_connection():
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")  # Kích hoạt ràng buộc khóa ngoại trong SQLite
    return con

def init_db():
    con = get_db_connection()
    cursor = con.cursor()
    
    print("Đang khởi tạo các bảng trong cơ sở dữ liệu mới...")

    # =============== 1. Bảng Users ===============
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
    
    # =============== 2. Bảng Refresh Tokens ===============
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
    
    # =============== 3. Bảng Problems ===============
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
        checker_path TEXT,
        source TEXT,
        input_folder_path TEXT,    
        output_folder_path TEXT, 
        input_zip_path TEXT,        -- Đường dẫn file zip input nén đưa trực tiếp vào bảng
        output_zip_path TEXT,       -- Đường dẫn file zip output nén đưa trực tiếp vào bảng
        is_public INTEGER NOT NULL DEFAULT 0,
        request_status TEXT NOT NULL DEFAULT 'NONE',  -- NONE, PENDING, APPROVED, REJECTED
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (author_id) REFERENCES users(id) ON DELETE SET NULL
    );
    """)
    
    # =============== 4. Bảng Submissions ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        problem_id INTEGER NOT NULL,
        submitted_code TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',  -- pending, accepted, wrong_answer, runtime_error, time_limit_exceeded
        score INTEGER DEFAULT 0,
        test_results TEXT,  -- Dữ liệu JSON lưu thông tin chi tiết từng testcase
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
    );
    """)
    
    # =============== 5. Bảng Draft Problem Sessions ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS draft_problem_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_name TEXT,
        repository_url TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        problems_json TEXT NOT NULL,
        research_title TEXT,                      -- Tích hợp trực tiếp cột tiêu đề nghiên cứu
        num_test_cases INTEGER DEFAULT 3,         -- Tích hợp trực tiếp số lượng testcases
        status TEXT NOT NULL DEFAULT 'draft',     -- 'draft' hoặc 'finalized'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # =============== 6. Bảng Roadmaps ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS roadmaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        repository_url TEXT NOT NULL,
        level TEXT NOT NULL,                      -- easy, medium, hard
        user_note TEXT,
        framework TEXT,                           -- pytorch, tensorflow, v.v.
        num_test_cases INTEGER DEFAULT 3,         -- Tích hợp trực tiếp số lượng cấu hình testcases
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== 7. Bảng Roadmap Problems (Nâng cấp toàn bộ cột nháp) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS roadmap_problems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_id INTEGER NOT NULL,
        problem_id INTEGER,                       -- Cho phép NULL khi bước này chưa được bấm 'Save to Problem'
        name TEXT NOT NULL,
        description TEXT,                         -- Lưu trữ mô tả nháp do AI sinh ra
        target_module TEXT,                       -- Đường dẫn module đích trong repository
        order_index INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',   -- Trạng thái: 'pending' (chưa tạo), 'generated' (đã sinh nháp), 'saved' (đã lưu chính thức)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE SET NULL
    );
    """)
    
    # =============== 8. Bảng Test Runs ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        submission_id INTEGER,
        problem_id INTEGER NOT NULL,
        testcase_index INTEGER NOT NULL,
        status TEXT NOT NULL,
        user_output TEXT,
        expected_output TEXT,
        error_message TEXT,
        elapsed_ms INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
    );
    """)
    
    con.commit()
    con.close()
    print("Khởi tạo toàn bộ các bảng cơ sở dữ liệu thành công.")

if __name__ == "__main__":
    # Tiến hành xóa cơ sở dữ liệu cũ nếu tồn tại để tạo mới hoàn toàn sạch sẽ
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Đã xóa file cơ sở dữ liệu cũ tại: {db_path}")
        except Exception as e:
            print(f"Không thể xóa file cũ tự động: {str(e)}. Hãy chắc chắn tiến trình khác đã đóng kết nối.")
            
    init_db()