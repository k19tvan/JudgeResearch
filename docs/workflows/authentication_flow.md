# Authentication Flow

## Overview
This workflow describes how a user registers, logs in, and authenticates API requests via JWT.

## Step-by-Step Execution
1. **User Action**: The End User opens the web client and navigates to the Login/Register component.
2. **Credential Transmission**: The UI calls `api.js` passing the plaintext credentials.
3. **Backend Validation**: `backend/main.py` routes the payload to `auth.py`. Handlers inspect the SQLite database for existing user records.
4. **Hashing**: If registering, `hash_password` converts string to hash and stores it. If logging in, `verify_password` validates the hash.
5. **Token Generation**: On success, `create_access_token` generates a time-restricted JWT string.
6. **Session Delivery**: Returned to the Frontend, which stores tokens, user id, role, username, and the current account avatar URL in localStorage.
7. **Subsequent Actions**: The frontend router interceptors pass the `Authorization` header on all protected endpoint requests.

## Related Workflows
- [Account Update Flow](account_update_flow.md)
- [Admin Account Management Flow](admin_account_management_flow.md)

# Updates

## 2026-06-01

### Change Summary
- Documented registration validation rules for required fields, username/email uniqueness, and password constraints.
- Noted user-facing registration outcomes and error messaging expectations.
- Account update requests now use the bearer access token to verify that the logged-in user owns the profile being changed.

### Files Modified
- backend/main.py
- backend/auth.py
- frontend/src/components/Register.jsx
- backend/main.py
- frontend/src/api.js

### Impact
- Registration enforces password rules (longer than 8 characters, digit required, no whitespace) and surfaces specific validation errors.
- Profile update and deactivation routes reject missing, expired, invalid, or cross-account tokens.

## 2026-06-02

### Change Summary
- Documented disabled-account authentication behavior for admin account management.

### Files Modified
- backend/main.py

### Impact
- Disabled accounts cannot log in or refresh access tokens.

## 2026-06-02 (Update 2)

### Change Summary
- Added avatar URL to the successful login session payload.
- Documented frontend clearing of stale avatar state when a logged-in account has no avatar or exits the session.

### Files Modified
- backend/main.py
- frontend/src/components/Login.jsx
- frontend/src/components/Home.jsx
- frontend/src/components/tabs/ProfileTab.jsx

### Impact
- Switching accounts no longer shows the previous account's avatar before the Profile tab is opened.
