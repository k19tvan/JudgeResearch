# Database Module

## Purpose
Abastracts and manages the native SQLite storage for the JudgeResearch platform, keeping track of users, problem sets, test cases, and evaluation results.

## Responsibilities
- Provide a bootstrap mechanism to generate tables safely.
- Abstract the connection interface to prevent connection locking.
- House initial schema shapes.

## Public Interfaces
- `initialize_database.py` script.
- Exposed connection pools or session generators in `__init__.py`.

## Dependencies
- OS File System for the `.db` SQLite artefact.

## Dependents
- Backend Module (calls queries to persist problems and users).

## Internal Design
Procedural. The script checks for the existence of the DB file. If missing, it creates standard SQL tables (Users, Problems, Submissions, Configs) using embedded DDL strings.

## Important Files
- [src_database_initialize.md](../files/src_database_initialize.md)

## Common Modification Tasks
- **How to add a new Table**: Add the `CREATE TABLE ...` command to `initialize_database.py`.
- **How to alter existing schema**: Right now, there is no automatic migration (like Alembic). Developers must write local update statements or reset their database for schema modifications.
