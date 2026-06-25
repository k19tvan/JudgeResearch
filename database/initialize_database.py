import sqlite3
import os
import argparse
from dotenv import load_dotenv
from backend.auth import hash_password

db_dir = "database"
db_name = "database.db"
db_path = os.path.join(db_dir, db_name)

# Đảm bảo thư mục lưu trữ cơ sở dữ liệu tồn tại
os.makedirs(db_dir, exist_ok=True)
load_dotenv()

def get_db_connection():
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")  # Kích hoạt ràng buộc khóa ngoại trong SQLite
    return con

# ================= HÀM DI TRÚ (MIGRATION) CHO DATABASE ĐÃ TỒN TẠI =================
def ensure_schema_migrations():
    """
    Tự động kiểm tra và thêm các cột mới vào các bảng nếu chạy trên database cũ 
    để đảm bảo tính tương thích và giữ nguyên dữ liệu hiện có.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 1. Kiểm tra và bổ sung cột cho bảng draft_problem_sessions
        cursor.execute("PRAGMA table_info(draft_problem_sessions)")
        draft_columns = {row[1] for row in cursor.fetchall()}
        if draft_columns:
            if "research_title" not in draft_columns:
                cursor.execute("ALTER TABLE draft_problem_sessions ADD COLUMN research_title TEXT")
            if "error_message" not in draft_columns:
                cursor.execute("ALTER TABLE draft_problem_sessions ADD COLUMN error_message TEXT")

        # 2. Kiểm tra và bổ sung cột cho bảng roadmap_problems
        cursor.execute("PRAGMA table_info(roadmap_problems)")
        rp_columns = {row[1] for row in cursor.fetchall()}
        if rp_columns:
            if "error_message" not in rp_columns:
                cursor.execute("ALTER TABLE roadmap_problems ADD COLUMN error_message TEXT")

        # 3. Kiểm tra và bổ sung cột cho bảng tickets
        cursor.execute("PRAGMA table_info(tickets)")
        ticket_columns = {row[1] for row in cursor.fetchall()}
        if ticket_columns and "image_url" not in ticket_columns:
            cursor.execute("ALTER TABLE tickets ADD COLUMN image_url TEXT")

        # 4. Kiểm tra và bổ sung cột cho bảng ticket_replies
        cursor.execute("PRAGMA table_info(ticket_replies)")
        reply_columns = {row[1] for row in cursor.fetchall()}
        if reply_columns and "image_url" not in reply_columns:
            cursor.execute("ALTER TABLE ticket_replies ADD COLUMN image_url TEXT")
            
        conn.commit()
        print("Đã hoàn tất việc kiểm tra và đồng bộ cấu trúc database cũ (nếu có).")
    except Exception as e:
        print(f"Lỗi khi thực hiện di trú cấu trúc bảng: {str(e)}")
    finally:
        conn.close()


# ================= HÀM KHỞI TẠO TOÀN BỘ DATABASE MỚI =================
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
    
    # =============== 5. Bảng Draft Problem Sessions (Đã cập nhật tích hợp error_message) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS draft_problem_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_name TEXT,
        repository_url TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        problems_json TEXT NOT NULL,
        research_title TEXT,                      -- Tiêu đề nghiên cứu
        num_test_cases INTEGER DEFAULT 3,         -- Số lượng testcases cấu hình
        status TEXT NOT NULL DEFAULT 'draft',     -- 'processing', 'draft', 'failed', hoặc 'finalized'
        error_message TEXT,                       -- Vết lỗi chi tiết nếu AI sinh thất bại
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
        num_test_cases INTEGER DEFAULT 3,         -- Số lượng cấu hình testcases
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== 7. Bảng Roadmap Problems (Đã cập nhật tích hợp error_message) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS roadmap_problems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_id INTEGER NOT NULL,
        problem_id INTEGER,                       -- Cho phép NULL khi bước này chưa được lưu chính thức
        name TEXT NOT NULL,
        description TEXT,                         -- Mô tả nháp do AI sinh ra
        target_module TEXT,                       -- Đường dẫn module đích trong repository
        order_index INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',   -- Trạng thái: 'pending', 'generating', 'generated', 'saved', 'failed'
        error_message TEXT,                       -- Lưu log lỗi biên dịch/runtime của tệp solution sinh nháp
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

    # =============== 9. Bảng Blogs (Bài viết chia sẻ) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS blogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        author_id INTEGER NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (author_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== 10. Bảng Comments (Hỗ trợ phân nhánh) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        problem_id INTEGER,      -- Trống nếu là bình luận của Blog
        blog_id INTEGER,         -- Trống nếu là thảo luận của Problem
        parent_id INTEGER,       -- Trỏ tới bình luận gốc để phân nhánh
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE,
        FOREIGN KEY (blog_id) REFERENCES blogs(id) ON DELETE CASCADE,
        FOREIGN KEY (parent_id) REFERENCES comments(id) ON DELETE CASCADE
    );
    """)

    # =============== 11. Bảng Votes (Upvote/Downvote bài viết & bình luận) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS votes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        blog_id INTEGER,         -- Upvote/downvote cho Blog
        comment_id INTEGER,      -- Upvote/downvote cho Comment
        vote_type INTEGER NOT NULL, -- 1 (Upvote) hoặc -1 (Downvote)
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, blog_id),
        UNIQUE(user_id, comment_id),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (blog_id) REFERENCES blogs(id) ON DELETE CASCADE,
        FOREIGN KEY (comment_id) REFERENCES comments(id) ON DELETE CASCADE
    );
    """)

    # =============== 12. Bảng Tickets (Đã cập nhật tích hợp image_url) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open', -- 'open' hoặc 'resolved'
        image_url TEXT,                      -- Danh sách đường dẫn ảnh đính kèm (Lưu dạng chuỗi JSON)
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== 13. Bảng Ticket Replies (Đã cập nhật tích hợp image_url) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticket_replies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        image_url TEXT,                      -- Ảnh đính kèm trong phản hồi (Lưu dạng chuỗi JSON)
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== 14. Bảng Solution Proposals (Đề xuất lời giải mẫu mới) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS solution_proposals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id INTEGER NOT NULL,
        contributor_id INTEGER NOT NULL,
        proposed_code TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING, APPROVED, REJECTED
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE,
        FOREIGN KEY (contributor_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    con.commit()

    # Thêm dữ liệu quản trị viên và tài khoản mẫu hữu ích
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        initial_admin_password = os.getenv("INITIAL_ADMIN_PASSWORD")
        if not initial_admin_password:
            raise RuntimeError("INITIAL_ADMIN_PASSWORD must be configured before creating the initial admin account")

        cursor.execute("""
            INSERT INTO users (username, password_hash, display_name, email, role, status)
            VALUES
            ('admin', ?, 'System Admin', 'admin@judgeresearch.com', 'admin', 'active')
        """, (hash_password(initial_admin_password),))
        con.commit()

        cursor.execute("SELECT id FROM users WHERE username='admin'")
        adm_id = cursor.fetchone()[0]

        cursor.execute("""
            INSERT INTO blogs (title, content, author_id)
            VALUES 
            ('Optimizing PyTorch Models in Production Environment', 'Deploying models efficiently requires a good grasp of compiler optimizations. In PyTorch, using torch.compile() introduces graph-level optimizations that can speed up inference by up to 2x.', ?),
            ('A Practical Guide to Learning Rate Scheduling', 'Learning rate decay schedules are highly important to ensure convergence without overfitting. Explore CosineAnnealingLR and ReduceLROnPlateau.', ?)
        """, (adm_id, adm_id))
        con.commit()

    con.close()
    print("Khởi tạo toàn bộ các bảng cơ sở dữ liệu thành công.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize or migrate the JudgeResearch database.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing database before initializing.")
    args = parser.parse_args()

    if args.reset and os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Removed existing database at: {db_path}")
        except Exception as e:
            print(f"Could not remove existing database: {str(e)}")

    init_db()
    ensure_schema_migrations()
