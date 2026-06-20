import sqlite3
import os
import argparse
from dotenv import load_dotenv
from backend.auth import hash_password

db_dir = "database"
db_name = "database.db"
db_path = os.path.join(db_dir, db_name)

# Äáº£m báº£o thÆ° má»¥c lÆ°u trá»¯ cÆ¡ sá»Ÿ dá»¯ liá»‡u tá»“n táº¡i
os.makedirs(db_dir, exist_ok=True)
load_dotenv()

def get_db_connection():
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")  # KĂ­ch hoáº¡t rĂ ng buá»™c khĂ³a ngoáº¡i trong SQLite
    return con

# ================= HĂ€M DI TRĂ (MIGRATION) CHO DATABASE ÄĂƒ Tá»’N Táº I =================
def ensure_schema_migrations():
    """
    Tá»± Ä‘á»™ng kiá»ƒm tra vĂ  thĂªm cĂ¡c cá»™t má»›i vĂ o cĂ¡c báº£ng náº¿u cháº¡y trĂªn database cÅ© 
    Ä‘á»ƒ Ä‘áº£m báº£o tĂ­nh tÆ°Æ¡ng thĂ­ch vĂ  giá»¯ nguyĂªn dá»¯ liá»‡u hiá»‡n cĂ³.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 1. Kiá»ƒm tra vĂ  bá»• sung cá»™t cho báº£ng draft_problem_sessions
        cursor.execute("PRAGMA table_info(draft_problem_sessions)")
        draft_columns = {row[1] for row in cursor.fetchall()}
        if draft_columns:
            if "research_title" not in draft_columns:
                cursor.execute("ALTER TABLE draft_problem_sessions ADD COLUMN research_title TEXT")
            if "error_message" not in draft_columns:
                cursor.execute("ALTER TABLE draft_problem_sessions ADD COLUMN error_message TEXT")

        # 2. Kiá»ƒm tra vĂ  bá»• sung cá»™t cho báº£ng roadmap_problems
        cursor.execute("PRAGMA table_info(roadmap_problems)")
        rp_columns = {row[1] for row in cursor.fetchall()}
        if rp_columns:
            if "error_message" not in rp_columns:
                cursor.execute("ALTER TABLE roadmap_problems ADD COLUMN error_message TEXT")

        # 3. Kiá»ƒm tra vĂ  bá»• sung cá»™t cho báº£ng tickets
        cursor.execute("PRAGMA table_info(tickets)")
        ticket_columns = {row[1] for row in cursor.fetchall()}
        if ticket_columns and "image_url" not in ticket_columns:
            cursor.execute("ALTER TABLE tickets ADD COLUMN image_url TEXT")

        # 4. Kiá»ƒm tra vĂ  bá»• sung cá»™t cho báº£ng ticket_replies
        cursor.execute("PRAGMA table_info(ticket_replies)")
        reply_columns = {row[1] for row in cursor.fetchall()}
        if reply_columns and "image_url" not in reply_columns:
            cursor.execute("ALTER TABLE ticket_replies ADD COLUMN image_url TEXT")
            
        conn.commit()
        print("ÄĂ£ hoĂ n táº¥t viá»‡c kiá»ƒm tra vĂ  Ä‘á»“ng bá»™ cáº¥u trĂºc database cÅ© (náº¿u cĂ³).")
    except Exception as e:
        print(f"Lá»—i khi thá»±c hiá»‡n di trĂº cáº¥u trĂºc báº£ng: {str(e)}")
    finally:
        conn.close()


# ================= HĂ€M KHá»I Táº O TOĂ€N Bá»˜ DATABASE Má»I =================
def init_db():
    con = get_db_connection()
    cursor = con.cursor()
    
    print("Äang khá»Ÿi táº¡o cĂ¡c báº£ng trong cÆ¡ sá»Ÿ dá»¯ liá»‡u má»›i...")

    # =============== 1. Báº£ng Users ===============
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
    
    # =============== 2. Báº£ng Refresh Tokens ===============
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
    
    # =============== 3. Báº£ng Problems ===============
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
        input_zip_path TEXT,        -- ÄÆ°á»ng dáº«n file zip input nĂ©n Ä‘Æ°a trá»±c tiáº¿p vĂ o báº£ng
        output_zip_path TEXT,       -- ÄÆ°á»ng dáº«n file zip output nĂ©n Ä‘Æ°a trá»±c tiáº¿p vĂ o báº£ng
        is_public INTEGER NOT NULL DEFAULT 0,
        request_status TEXT NOT NULL DEFAULT 'NONE',  -- NONE, PENDING, APPROVED, REJECTED
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (author_id) REFERENCES users(id) ON DELETE SET NULL
    );
    """)
    
    # =============== 4. Báº£ng Submissions ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        problem_id INTEGER NOT NULL,
        submitted_code TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',  -- pending, accepted, wrong_answer, runtime_error, time_limit_exceeded
        score INTEGER DEFAULT 0,
        test_results TEXT,  -- Dá»¯ liá»‡u JSON lÆ°u thĂ´ng tin chi tiáº¿t tá»«ng testcase
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
    );
    """)
    
    # =============== 5. Báº£ng Draft Problem Sessions (ÄĂ£ cáº­p nháº­t tĂ­ch há»£p error_message) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS draft_problem_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_name TEXT,
        repository_url TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        problems_json TEXT NOT NULL,
        research_title TEXT,                      -- TiĂªu Ä‘á» nghiĂªn cá»©u
        num_test_cases INTEGER DEFAULT 3,         -- Sá»‘ lÆ°á»£ng testcases cáº¥u hĂ¬nh
        status TEXT NOT NULL DEFAULT 'draft',     -- 'processing', 'draft', 'failed', hoáº·c 'finalized'
        error_message TEXT,                       -- Váº¿t lá»—i chi tiáº¿t náº¿u AI sinh tháº¥t báº¡i
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # =============== 6. Báº£ng Roadmaps ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS roadmaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        repository_url TEXT NOT NULL,
        level TEXT NOT NULL,                      -- easy, medium, hard
        user_note TEXT,
        framework TEXT,                           -- pytorch, tensorflow, v.v.
        num_test_cases INTEGER DEFAULT 3,         -- Sá»‘ lÆ°á»£ng cáº¥u hĂ¬nh testcases
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== 7. Báº£ng Roadmap Problems (ÄĂ£ cáº­p nháº­t tĂ­ch há»£p error_message) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS roadmap_problems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadmap_id INTEGER NOT NULL,
        problem_id INTEGER,                       -- Cho phĂ©p NULL khi bÆ°á»›c nĂ y chÆ°a Ä‘Æ°á»£c lÆ°u chĂ­nh thá»©c
        name TEXT NOT NULL,
        description TEXT,                         -- MĂ´ táº£ nhĂ¡p do AI sinh ra
        target_module TEXT,                       -- ÄÆ°á»ng dáº«n module Ä‘Ă­ch trong repository
        order_index INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',   -- Tráº¡ng thĂ¡i: 'pending', 'generating', 'generated', 'saved', 'failed'
        error_message TEXT,                       -- LÆ°u log lá»—i biĂªn dá»‹ch/runtime cá»§a tá»‡p solution sinh nhĂ¡p
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE SET NULL
    );
    """)
    
    # =============== 8. Báº£ng Test Runs ===============
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

    # =============== 9. Báº£ng Blogs (BĂ i viáº¿t chia sáº») ===============
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

    # =============== 10. Báº£ng Comments (Há»— trá»£ phĂ¢n nhĂ¡nh) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        problem_id INTEGER,      -- Trá»‘ng náº¿u lĂ  bĂ¬nh luáº­n cá»§a Blog
        blog_id INTEGER,         -- Trá»‘ng náº¿u lĂ  tháº£o luáº­n cá»§a Problem
        parent_id INTEGER,       -- Trá» tá»›i bĂ¬nh luáº­n gá»‘c Ä‘á»ƒ phĂ¢n nhĂ¡nh
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE,
        FOREIGN KEY (blog_id) REFERENCES blogs(id) ON DELETE CASCADE,
        FOREIGN KEY (parent_id) REFERENCES comments(id) ON DELETE CASCADE
    );
    """)

    # =============== 11. Báº£ng Votes (Upvote/Downvote bĂ i viáº¿t & bĂ¬nh luáº­n) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS votes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        blog_id INTEGER,         -- Upvote/downvote cho Blog
        comment_id INTEGER,      -- Upvote/downvote cho Comment
        vote_type INTEGER NOT NULL, -- 1 (Upvote) hoáº·c -1 (Downvote)
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, blog_id),
        UNIQUE(user_id, comment_id),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (blog_id) REFERENCES blogs(id) ON DELETE CASCADE,
        FOREIGN KEY (comment_id) REFERENCES comments(id) ON DELETE CASCADE
    );
    """)

    # =============== 12. Báº£ng Tickets (ÄĂ£ cáº­p nháº­t tĂ­ch há»£p image_url) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open', -- 'open' hoáº·c 'resolved'
        image_url TEXT,                      -- Danh sĂ¡ch Ä‘Æ°á»ng dáº«n áº£nh Ä‘Ă­nh kĂ¨m (LÆ°u dáº¡ng chuá»—i JSON)
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== 13. Báº£ng Ticket Replies (ÄĂ£ cáº­p nháº­t tĂ­ch há»£p image_url) ===============
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticket_replies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        image_url TEXT,                      -- áº¢nh Ä‘Ă­nh kĂ¨m trong pháº£n há»“i (LÆ°u dáº¡ng chuá»—i JSON)
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # =============== 14. Báº£ng Solution Proposals (Äá» xuáº¥t lá»i giáº£i máº«u má»›i) ===============
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

    # ThĂªm dá»¯ liá»‡u quáº£n trá»‹ viĂªn vĂ  tĂ i khoáº£n máº«u há»¯u Ă­ch
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
    print("Khá»Ÿi táº¡o toĂ n bá»™ cĂ¡c báº£ng cÆ¡ sá»Ÿ dá»¯ liá»‡u thĂ nh cĂ´ng.")

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
