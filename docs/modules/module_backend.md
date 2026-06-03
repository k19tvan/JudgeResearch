# Backend Module

## Purpose
The primary backend service handling operations logic for JudgeResearch. Facilitates problem hosting, solving, evaluation workflows, and security.

## Responsibilities
- Routing API HTTP requests from the frontend.
- Connecting to the SQLite database via `database` module.
- Securing routes using JWT authentication.
- Enforcing account update ownership, profile validation, and deactivation cleanup.
- Enforcing admin-only user/account management, including role and status changes.
- Triggering prompt requests and executing tests against solutions using isolated sandbox processing.
- File system management for `storage/problems/` (unzipping and validating test cases).

## Public Interfaces
- FastAPI application exposed via HTTP (default port 21081).
- Endpoints primarily defined in `main.py`.
- Authentication endpoints in `auth.py`.

## Dependencies
- `fastapi`, `uvicorn` for the web framework.
- `database` module for data access.
- `prompts` for AI template configurations.
- Dynamic module imports for executing untrusted user code via `subprocess`.

## Dependents
- Frontend Module.

## Internal Design
It uses a modularized logic design. `main.py` bootstraps the FastAPI app and includes route handlers. `auth.py` abstracts password hashing and token generation. Code execution is delegated to `sandbox.py`, and testcase file operations (zip/extract) natively to `file_manager.py`. Data operations are done synchronously or asynchronously depending on db driver capability.

## Important Files
- [src_backend_main.md](../files/src_backend_main.md)
- [src_backend_auth.md](../files/src_backend_auth.md)
- [src_backend_sandbox.md](../files/src_backend_sandbox.md)
- [src_backend_file_manager.md](../files/src_backend_file_manager.md)

# Updates

## 2026-06-01

### Change Summary
- Added backend responsibility for account update validation and account deactivation cleanup.

### Files Modified
- backend/main.py
- backend/auth.py

### Impact
- Backend account update routes now require authenticated ownership and enforce the revised username/password/email rules.

## 2026-06-02

### Change Summary
- Added backend responsibility for admin-only account management.
- Added disabled-account checks for login and token refresh.

### Files Modified
- backend/main.py

### Impact
- Active admins can manage account information, roles, and status while disabled accounts cannot authenticate.

## Common Modification Tasks
- **How to add a new API route**: Define a new `@app.get(...)` or `@app.post(...)` in `backend/main.py`.
- **How to modify the auth flow**: Update token decoding logic in `backend/auth.py`.
