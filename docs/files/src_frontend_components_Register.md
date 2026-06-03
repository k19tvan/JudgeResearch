# frontend/src/components/Register.jsx

## Original Path
`frontend/src/components/Register.jsx`

## Purpose
Registration UI for new users. Collects account details, validates input, and submits to the backend registration endpoint.

## Key Behaviors
- Client-side validation for required fields and password rules.
- Submits registration data via the API client and redirects to login on success.
- Provides a Back button to return to the login screen without saving.

## Related Files
- [src_frontend_api.md](src_frontend_api.md)
- [src_backend_main.md](src_backend_main.md)

# Updates

## 2026-06-01

### Change Summary
- Added required-field checks and password rule validation with field highlighting.
- Aligned success messaging and added Back navigation to login.
- Updated password length messaging to match the stricter account rule requiring more than 8 characters.

### Files Modified
- frontend/src/components/Register.jsx
- backend/auth.py

### Impact
- Users receive immediate feedback for invalid registration inputs before account creation.
- Client-side registration password validation now matches the backend's shared password validation helper.
