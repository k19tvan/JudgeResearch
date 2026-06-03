# database/initialize_database.py

## Original Path
`database/initialize_database.py`

## Purpose
A one-time utility script designed to set up the structural schema of the platform's SQLite database instance.

## Imports
- `sqlite3`: Native python database bindings.

## Functions
- Generates tables: Users, Problems, Submissions.

## Execution Notes
Run from terminal as standard python script: `python -m database.initialize_database`. It should safely skip if data exists, avoiding destruction of persistent data.

## Modification Risks
Renaming tables requires manual back-migration for existing development setups. Changing column types on SQLite can be difficult due to limited `ALTER TABLE` support.

## Related Files
- [src_backend_main.md](src_backend_main.md)
