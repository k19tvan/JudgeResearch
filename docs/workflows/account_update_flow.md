# Account Update Flow

## Use Case 1.3 - Account Update (Profile Editing)

### Objective
Allow a logged-in user to review and update their own account information, with validation for usernames, passwords, email addresses, and account deactivation.

### Actors
- Logged-in end user.
- Backend API.
- SQLite database.

### Preconditions
- The user is logged in and has a valid bearer access token.
- The account being viewed, updated, or deactivated exists in the database.
- The authenticated user id in the token must match the account being updated. Older tokens without user id may fall back to username until refreshed.

### Main Flow
1. The user logs into the system and opens the Profile screen.
2. The user selects **Edit Profile**.
3. The system displays the current username, display name, email, avatar, and an optional new-password field.
4. The user updates the desired fields.
5. The user clicks **Update**.
6. The frontend validates required fields, username format, password rules, and email format.
7. The backend verifies the user is authenticated as the account owner.
8. The backend validates the submitted fields against database-level business rules.
9. The backend updates the user information in the database.
10. The system displays **Update successful**.

### Alternative Flows
- **Back without saving**: If the user clicks **Back**, the system exits edit mode, restores the current saved profile values, and does not call the update API.
- **Invalid update data**: If any frontend or backend validation fails, the system displays a specific error message and aborts the update.
- **Missing authentication**: If the user is not logged in, the backend rejects the request and no data is changed.
- **Expired access token**: If the access token has expired but the refresh token is still valid, the frontend refreshes the access token and retries the account request once before showing an error.
- **Account not found**: If the account no longer exists, the backend returns an account-not-found error.
- **Account deactivation**: If the user selects **Deactivate Account** and confirms, the backend permanently deletes the account and associated data, then the frontend clears local session data and returns to login.

### Validation Rules
- Required profile fields must not be empty: username, display name, and email.
- Username must be 3-32 characters.
- Username may contain only letters, numbers, dots, underscores, and dashes.
- Username must not duplicate any other username.
- Username must not duplicate any display name, including the user's effective display name after the submitted update.
- Username is checked against the configured `USERNAME_BLOCKLIST` terms for community-standard violations.
- Password is optional during profile update. When supplied, it must be longer than 8 characters, contain at least one digit, and contain no whitespace.
- Email must match standard email format and must not duplicate another account email.
- Email ownership is enforced only by requiring the authenticated account owner to submit the change; external email-verification is not currently implemented.
- Avatar is optional. When supplied, it must be selected from local files. The system must validate that the file type is an image (JPG, PNG, WEBP) and that the file size is less than or equal to 5MB.

### Postconditions
- On successful update, the database stores the new account values and updates `updated_at`.
- On successful update, the backend returns a refreshed access token so future edits still authorize after a username change.
- On failed validation, no account fields are changed.
- On confirmed deactivation, the account can no longer be used because the user row, refresh tokens, submissions, drafts, roadmaps, and authored problems are deleted.

### Original Specification Gaps
- The original flow did not define token/owner authorization for profile updates.
- The original frontend profile screen displayed account details but did not provide edit controls.
- The original frontend API did not expose account deactivation.
- Password length wording was inconsistent: one section said "longer than 8 characters" while another said "at least 8 characters"; implementation now follows the stricter "longer than 8 characters" rule.
- Legitimate personal email ownership cannot be proven without an email verification workflow, which is not present in this system.

# Updates

## 2026-06-01

### Change Summary
- Added the revised Use Case 1.3 specification for profile editing and account deactivation.
- Documented validation rules and identified original specification gaps.

### Files Modified
- docs/workflows/account_update_flow.md
- backend/main.py
- backend/auth.py
- frontend/src/api.js
- frontend/src/components/tabs/ProfileTab.jsx

### Impact
- Future account update changes have a dedicated workflow document that maps product behavior to backend and frontend implementation.

## 2026-06-02

### Change Summary
- Fixed repeat profile updates after username changes by treating signed `user_id` as the durable account owner identity.
- Added refreshed access tokens to successful account update responses.
- Added avatar URL editing to the account update flow.

### Files Modified
- docs/workflows/account_update_flow.md
- backend/main.py
- frontend/src/api.js
- frontend/src/components/tabs/ProfileTab.jsx

### Impact
- Users can continue updating their own account after changing username, and can update or clear their avatar URL.

## 2026-06-02 (Update 2)

### Change Summary
- Refactored avatar upload mechanism from a URL text input to local file upload.
- Integrated file type and size validation for avatars.
- Handled multipart/form-data for profile updating.

### Files Modified
- docs/workflows/account_update_flow.md
- backend/main.py
- frontend/src/api.js
- frontend/src/components/tabs/ProfileTab.jsx

### Impact
- Avatars are securely validated (must be image files under 5MB) and stored locally on the server.

## 2026-06-02 (Update 3)

### Change Summary
- Added frontend retry behavior for expired access tokens during profile fetch, update, and deactivation.

### Files Modified
- docs/workflows/account_update_flow.md
- frontend/src/api.js

### Impact
- Users can return to the Profile tab after access-token expiry without losing access to the profile editing UI, as long as their refresh token is still valid.
