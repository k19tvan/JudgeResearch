# backend/auth.py

## Original Path
`backend/auth.py`

## Purpose
Securing the API through password hashing, token generation (JWT), and route verification contexts.

## Imports
- `passlib`: For hashing and verifying passwords securely (commonly `bcrypt`).
- `jose`: For evaluating JSON Web Tokens.
- `fastapi.security`: To implement `OAuth2PasswordBearer`.

## Functions
- `hash_password(plain_password)`: Converts string to a stored hash.
- `verify_password(plain_password, hashed_password)`: Compares inputs.
- `create_access_token(data, expires_delta)`: Packages user context into an encoded JWT via secret secret.
- `get_current_user(...)`: FastAPI dependency that intercepts requests, checks tokens, and yields user states.
- `validate_username(username)`: Checks username length, allowed characters, and configured community blocklist terms.
- `validate_password(password)`: Checks password length, digit presence, and whitespace restrictions.

## Execution Notes
Make sure `SECRET_KEY` is loaded from the environment to prevent token interception.

## Modification Risks
Weakening the hashing algorithm risks credential compromise. Invalidating token expiration causes immediate forced logouts for end users.

## Related Files
- [src_backend_main.md](src_backend_main.md)

# Updates

## 2026-06-01

### Change Summary
- Added password validation helper for registration rules (length, digit required, no whitespace).
- Updated password validation to require passwords longer than 8 characters for the revised account update rule.
- Reused username validation for account updates, including format and community-standard blocklist checks.

### Files Modified
- backend/auth.py
- backend/main.py

### Impact
- Registration now has centralized password rule checks reused by the API layer.
- Account update uses the same validation helpers to keep frontend and backend account rules consistent.
