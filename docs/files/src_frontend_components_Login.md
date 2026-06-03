# frontend/src/components/Login.jsx

## Original Path
`frontend/src/components/Login.jsx`

## Purpose
Login UI for authenticating a user and initializing the frontend session in local storage.

## Key Behaviors
- Submits username and password through `loginUser`.
- Stores access token, refresh token, username, user id, role, and avatar URL on successful login.
- Removes `avatar_url` when the authenticated account has no avatar.
- Navigates to the dashboard after session state is written.
- Displays backend authentication errors.

## Related Files
- [src_frontend_api.md](src_frontend_api.md)
- [src_backend_main.md](src_backend_main.md)
- [src_frontend_components_Home.md](src_frontend_components_Home.md)

# Updates

## 2026-06-02

### Change Summary
- Added avatar URL session initialization and stale avatar cleanup during login.

### Files Modified
- frontend/src/components/Login.jsx
- backend/main.py
- frontend/src/components/Home.jsx

### Impact
- Switching accounts updates or clears the dashboard avatar immediately after login.
