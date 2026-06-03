# frontend/src/api.js

## Original Path
`frontend/src/api.js`

## Purpose
A centralized HTTP client mechanism (commonly wrapping `axios` or standard `fetch`) tailored to communicate with the FastAPI backend endpoint `localhost:21081`.

## Imports
- `axios` (or standard `fetch` wrappers).

## Execution Notes
Usually configures interceptors to automatically strap the `Authorization: Bearer <token>` to headers when available in local storage.

## Account Update Helpers
- `fetchUserProfile(userId)`: Fetches the logged-in account owner's profile with the bearer token.
- `updateUserProfile(userId, profileData)`: Sends profile changes to the backend update endpoint with the bearer token.
- `deactivateUserAccount(userId)`: Confirms permanent account deletion with the backend deactivation endpoint.
- Account profile helpers refresh the access token and retry once when the backend reports an expired/invalid access token or stale owner identity.

## Auth Helpers
- `loginUser(credentials)`: Authenticates a user and returns tokens plus session metadata, including `user_id`, `user_role`, and `avatar_url`.

## Admin Account Management Helpers
- `fetchManagedUsers(search)`: Fetches the admin-only account list, optionally filtered by search text.
- `fetchManagedUserDetails(userId)`: Fetches a selected account's details for admin editing.
- `updateManagedUser(userId, accountData)`: Updates display name, email, role, and status for a managed account.

## Modification Risks
Incorrect base URLs break communication. Stripping interceptors prevents authenticated routes from functioning.

## Related Files
- [src_frontend_App.md](src_frontend_App.md)
- [src_backend_main.md](src_backend_main.md)
- [src_frontend_components_tabs_AccountManagementTab.md](src_frontend_components_tabs_AccountManagementTab.md)

# Updates

## 2026-06-01

### Change Summary
- Added authenticated headers for account profile fetch/update requests.
- Added the account deactivation API helper.

### Files Modified
- frontend/src/api.js
- frontend/src/components/tabs/ProfileTab.jsx
- backend/main.py

### Impact
- Profile editing and deactivation now use the same logged-in user token expected by the backend Account Update flow.

## 2026-06-02

### Change Summary
- Added shared auth retry behavior for profile fetch, update, and deactivation.
- Retained multipart form submission for avatar uploads.

### Files Modified
- frontend/src/api.js

### Impact
- Expired access tokens no longer hide the Profile edit UI when a valid refresh token is available.

## 2026-06-02 (Update 2)

### Change Summary
- Added admin account-management API helpers for list, detail, and update operations.

### Files Modified
- frontend/src/api.js
- frontend/src/components/tabs/UsersTab.jsx

### Impact
- The Account Management tab can call the admin-only backend routes with existing bearer-token retry behavior.

## 2026-06-02 (Update 3)

### Change Summary
- Updated account-management helper ownership to the new `AccountManagementTab.jsx` component.

### Files Modified
- frontend/src/api.js
- frontend/src/components/tabs/AccountManagementTab.jsx

### Impact
- API helper documentation now matches the separated Account Management tab.

## 2026-06-02 (Update 4)

### Change Summary
- Documented `avatar_url` in the login response consumed by `Login.jsx`.

### Files Modified
- frontend/src/api.js
- frontend/src/components/Login.jsx
- backend/main.py

### Impact
- Login/session documentation now matches the avatar initialization behavior.
