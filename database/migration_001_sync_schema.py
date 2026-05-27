"""
Migration 001: Synchronize database schema with design specification
This migration:
1. Renames research_roadmaps → roadmaps and updates fields
2. Renames research_problems → roadmap_problems and updates fields
3. Updates problems table to match design (rename approval_status logic)
4. Creates submissions table for live coding
5. Adds missing fields to users and refresh_tokens tables
6. Preserves draft_problem_sessions as intermediate table for iterative refinement
"""

import sqlite3
from datetime import datetime

def migrate_up(db_path: str = "database/database.db"):
    """Apply migration changes"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # ============ STEP 1: Add missing fields to existing tables ============
        print("[1/5] Updating users table...")
        cursor.execute("PRAGMA table_info(users)")
        user_columns = {row[1] for row in cursor.fetchall()}
        
        if "avatar_url" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
            print("  ✓ Added avatar_url column")
        
        # ============ STEP 2: Update refresh_tokens table ============
        print("[2/5] Updating refresh_tokens table...")
        cursor.execute("PRAGMA table_info(refresh_tokens)")
        token_columns = {row[1] for row in cursor.fetchall()}
        
        if "is_revoked" not in token_columns:
            cursor.execute("ALTER TABLE refresh_tokens ADD COLUMN is_revoked INTEGER DEFAULT 0")
            print("  ✓ Added is_revoked column")
        
        # ============ STEP 3: Update problems table ============
        print("[3/5] Updating problems table...")
        cursor.execute("PRAGMA table_info(problems)")
        problem_columns = {row[1] for row in cursor.fetchall()}
        
        # Rename approval_status to request_status if it exists
        if "approval_status" in problem_columns and "request_status" not in problem_columns:
            cursor.execute("ALTER TABLE problems RENAME COLUMN approval_status TO request_status")
            print("  ✓ Renamed approval_status → request_status")
        
        # Add missing fields
        if "author_id" not in problem_columns and "creator_id" in problem_columns:
            # Create author_id as alias for creator_id for now
            print("  ℹ Using creator_id as author_id (consider renaming in app code)")
        
        if "solution_path" not in problem_columns:
            cursor.execute("ALTER TABLE problems ADD COLUMN solution_path TEXT")
            print("  ✓ Added solution_path column")
        
        if "input_zip_url" not in problem_columns:
            cursor.execute("ALTER TABLE problems ADD COLUMN input_zip_url TEXT")
            print("  ✓ Added input_zip_url column")
        
        if "output_zip_url" not in problem_columns:
            cursor.execute("ALTER TABLE problems ADD COLUMN output_zip_url TEXT")
            print("  ✓ Added output_zip_url column")
        
        # Ensure request_status has correct default
        cursor.execute("""
            UPDATE problems 
            SET request_status = 'NONE' 
            WHERE request_status IS NULL OR request_status = ''
        """)
        
        # ============ STEP 4: Create submissions table ============
        print("[4/5] Creating submissions table...")
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
        print("  ✓ Created submissions table")
        
        # ============ STEP 5: Create/Update roadmaps table ============
        print("[5/5] Creating roadmaps table...")
        
        # Check if roadmaps table exists with correct schema
        cursor.execute("PRAGMA table_info(roadmaps)")
        existing_roadmap_cols = {row[1] for row in cursor.fetchall()}
        
        if not existing_roadmap_cols:
            # Create new roadmaps table with full schema
            cursor.execute("""
                CREATE TABLE roadmaps (
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
            print("  ✓ Created new roadmaps table")
            
            # Migrate data from research_roadmaps if it exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='research_roadmaps'")
            if cursor.fetchone():
                print("  ℹ Migrating data from research_roadmaps...")
                cursor.execute("""
                    INSERT INTO roadmaps (id, user_id, name, repository_url, status, created_at)
                    SELECT id, creator_id, title, repository_url, 'draft', created_at
                    FROM research_roadmaps
                """)
                print("  ✓ Migrated research_roadmaps data")
        
        # ============ STEP 6: Create/Update roadmap_problems table ============
        print("[6/6] Creating roadmap_problems table...")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='roadmap_problems'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE roadmap_problems (
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
            print("  ✓ Created new roadmap_problems table")
            
            # Migrate data from research_problems if it exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='research_problems'")
            if cursor.fetchone():
                print("  ℹ Migrating data from research_problems...")
                cursor.execute("""
                    INSERT INTO roadmap_problems (roadmap_id, problem_id, order_index, status)
                    SELECT research_id, problem_id, step_order, 'completed'
                    FROM research_problems
                """)
                print("  ✓ Migrated research_problems data")
        
        # ============ STEP 7: Update draft_problem_sessions if needed ============
        print("[7/7] Ensuring draft_problem_sessions is properly configured...")
        cursor.execute("PRAGMA table_info(draft_problem_sessions)")
        session_columns = {row[1] for row in cursor.fetchall()}
        
        if session_columns:  # Table exists
            if "research_title" in session_columns and "roadmap_name" not in session_columns:
                cursor.execute("ALTER TABLE draft_problem_sessions RENAME COLUMN research_title TO roadmap_name")
                print("  ✓ Renamed research_title → roadmap_name")
        
        conn.commit()
        print("\n✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {str(e)}")
        raise
    finally:
        conn.close()


def migrate_down(db_path: str = "database/database.db"):
    """Rollback migration - use with caution!"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("Rolling back migration...")
        
        # Drop new tables (keep them for safety - manual intervention recommended)
        print("⚠️  New tables (submissions, roadmaps, roadmap_problems) will be kept for data safety.")
        print("    Review and manually delete if needed after verifying backup exists.")
        
        conn.commit()
        print("\n⚠️  Rollback completed. Review changes manually.")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Rollback failed: {str(e)}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    
    db_path = "database/database.db"
    action = sys.argv[1] if len(sys.argv) > 1 else "up"
    
    print(f"Database migration script")
    print(f"Path: {db_path}\n")
    
    if action == "up":
        migrate_up(db_path)
    elif action == "down":
        migrate_down(db_path)
    else:
        print(f"Unknown action: {action}")
        print("Usage: python migration_001_sync_schema.py [up|down]")
        sys.exit(1)
