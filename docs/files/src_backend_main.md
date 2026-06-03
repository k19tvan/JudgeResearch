# backend/main.py

## Original Path
`backend/main.py`

## Purpose
Entry point for the JudgeResearch FastAPI application. It wires together HTTP endpoints, configures CORS, and abstracts requests to the problem evaluation layers.

## Imports
- `fastapi`: Core framework for REST endpoints, file uploads (`UploadFile`, `File`, `Form`).
- `database`: Used directly or indirectly to query/update user and problem data.
- `backend.auth`: Pulling authentication schemes and token definitions to guard routes.
- `backend.file_manager` & `backend.sandbox`: Delegates disk operation and code execution for problems.
- `prompts.prompt`: For AI integration into problem text generation.

## Classes
N/A (Primarily relies on functional route definitions).

## Functions
- `app.get("/problems")`: Fetches the paginated list of problems.
- `app.post("/submit")`: Evaluates standard code execution against test cases using `sandbox.execute_user_code`.
- `app.post("/problems/upload")` (or similar file endpoints): Delegates zip ingestions to `file_manager.py`.
- `app.post("/api/auth/login")`: Authenticates active accounts and returns tokens plus session metadata including role, user id, and avatar URL.
- `app.get("/api/users/{user_id}")`: Returns the authenticated account owner's profile for account update.
- `app.put("/api/users/{user_id}")`: Updates profile fields after ownership, username, password, and email validation.
- `app.post("/api/users/{user_id}/deactivate")`: Permanently deletes the authenticated user's account and associated data after confirmation.
- `app.get("/api/admin/users")`: Lists user accounts for active administrators.
- `app.get("/api/admin/users/{target_user_id}")`: Returns account details for active administrators.
- `app.put("/api/admin/users/{target_user_id}")`: Updates display name, email, role, and status for active administrators.
- *Dependencies*: Relies on python environments and `sqlite3` for fast storage bindings.

## Execution Notes
Run via `uvicorn backend.main:app --host 0.0.0.0 --port 21081 --reload`. Ensure `env` is configured properly.

## Modification Risks
Changing the `@app` router variables can break the Vite Frontend proxy if not aligned. Security risks exist if Auth middleware is bypassed.

## Related Files
- [src_backend_auth.md](src_backend_auth.md)
- [src_frontend_api.md](src_frontend_api.md)

# Updates

## 2026-06-01

### Change Summary
- Added required-field checks and password rule enforcement for registration.
- Split duplicate checks to return specific username/email errors and aligned success message.
- Required bearer-token ownership for account profile fetch, update, and deactivation.
- Enforced account-update username uniqueness against both usernames and display names.
- Added permanent deactivation cleanup for submissions, refresh tokens, drafts, roadmaps, authored problems, and the user row.

### Files Modified
- backend/main.py
- frontend/src/api.js
- frontend/src/components/tabs/ProfileTab.jsx

### Impact
- Registration rejects invalid inputs with specific error messages before account creation.
- Account update now aborts on invalid username, password, email, ownership, or deactivation confirmation checks.

## 2026-06-02

### Change Summary
- Added Use Case 1.4 admin account-management endpoints.
- Added active-admin authorization helper.
- Added disabled-account rejection during login and token refresh.

### Files Modified
- backend/main.py

### Impact
- Admin users can list, inspect, and update accounts while non-admin or disabled users cannot use account-management APIs.

## 2026-06-02 (Update 2)

### Change Summary
- Added `avatar_url` to the login response payload.

### Files Modified
- backend/main.py
- frontend/src/components/Login.jsx
- frontend/src/components/Home.jsx

### Impact
- The frontend can initialize the dashboard avatar from the authenticated account instead of stale local storage.

## 2026-06-03 (Update 3)

### Change Summary
- Added `storage/tickets` directory generation and `/tickets_media` static mount.
- Updated ticket creation endpoint to accept `multipart/form-data` and save images.
- Included `image_url` column in ticket SELECT queries.

### Files Modified
- backend/main.py

### Impact
- Ticket routes now securely accept and return image attachments.

## 2026-06-03 (Update 4)

### Change Summary
- Rewrote `@app.post("/api/tickets")` to parse `authorization: Header(None)` via `get_authenticated_identity`.
- Updated `create_ticket` to iterate over `Optional[List[UploadFile]]` and save paths via `json.dumps()`.

### Files Modified
- backend/main.py

### Impact
- Eliminates reliance on unverified frontend `user_id` payloads.
- Resolves Foreign Key constraint errors on ticket creation.

## 2026-06-03 (Update 5)

### Change Summary
- Rewrote `@app.post("/api/tickets/{ticket_id}/replies")` to parse `authorization: Header(None)` and process `Optional[List[UploadFile]]`.
- Updated schema via `ensure_schema_migrations` to add `image_url` to `ticket_replies`.
- Added `r.image_url` to the ticket details SELECT query.

### Impact
- Administrators and Users can seamlessly exchange screenshots back-and-forth natively within a ticket thread.

## 2026-06-03 (Update 6)

### Change Summary
- Added `PUT /api/tickets/{ticket_id}` and `DELETE /api/tickets/{ticket_id}`.
- Added `PUT /api/tickets/replies/{reply_id}` and `DELETE /api/tickets/replies/{reply_id}`.
- Refactored `POST /api/tickets/{ticket_id}/status` to strictly enforce `admin` role checks.

### Impact
- Image editing handles partial file retention securely without overwriting unmodified assets. 
- Prevents end-users from force-closing open investigations.

## 2026-06-03 (Update 7)

### Change Summary
- Added `status == "resolved"` checks to all ticket POST, PUT, and DELETE endpoints.
- Added cross-reference checks to reply PUT and DELETE endpoints to verify the parent ticket status.

### Impact
- Enforces data integrity on closed support tickets.

## 2026-06-03 (Update 8)

### Change Summary
- Removed `user_id` ownership isolation checks from `GET /api/tickets`, `GET /api/tickets/{ticket_id}`, and `POST /api/tickets/{ticket_id}/replies`.

### Impact
- Standard users now receive the full array of global tickets, allowing them to participate in public debugging/support discussions.