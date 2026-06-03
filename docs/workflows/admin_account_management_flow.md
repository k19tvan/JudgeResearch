# Admin Account Management Flow

## Use Case 1.4 - User/Account Management (Admin Only)

### Objective
Allow active administrators to view, search, inspect, and update user accounts, including display name, email, role, and account status.

### Actors
- Logged-in administrator.
- Backend API.
- SQLite database.

### Preconditions
- The acting user is logged in with a valid access token.
- The acting user exists in the database with `role = 'admin'` and `status = 'active'`.

### Main Flow
1. The administrator selects **Account Management** in the dashboard navigation.
2. The frontend requests the managed account list from `/api/admin/users`.
3. The backend validates the bearer token and confirms the acting user is an active admin.
4. The backend returns all user accounts, optionally filtered by search text.
5. The administrator selects an account from the list.
6. The frontend requests the selected account details from `/api/admin/users/{target_user_id}`.
7. The administrator edits display name, email, role, or status.
8. The administrator clicks **Save**.
9. The frontend validates required display name and email format.
10. The backend validates admin privileges, account existence, email format, duplicate email, role, and status.
11. The backend updates the account in the `users` table and returns **Update successful** with the updated account.
12. The frontend updates the list and detail panel.

### Alternative Flows
- **Cancel/Back**: The administrator clicks **Cancel** to discard unsaved form changes or **Back** to return to the account list state without saving.
- **Invalid update data**: If display name is empty, email is invalid/duplicated, role is unsupported, or status is unsupported, the backend aborts the update and the UI displays the error.
- **Unauthorized access**: If the acting user is not an active admin, the backend returns an admin-privilege error and no data is changed.
- **Disabled account login**: A disabled account cannot log in or refresh an access token.

### Validation Rules
- `display_name` is required when updated.
- `email` is required, must match standard email format, and must be unique across users.
- `role` must be one of `user`, `contributor`, or `admin`.
- `status` must be one of `active` or `disabled`.

### Postconditions
- On success, the selected account's personal information, role, and status are updated in the database.
- On failure, no account fields are changed and the UI displays a specific error message.

# Updates

## 2026-06-02

### Change Summary
- Added the admin-only account management workflow specification.

### Files Modified
- docs/workflows/admin_account_management_flow.md
- backend/main.py
- frontend/src/api.js
- frontend/src/components/Home.jsx
- frontend/src/components/tabs/UsersTab.jsx

### Impact
- Use Case 1.4 has a documented flow for admin account list, detail, update, activation, and disabling behavior.

## 2026-06-02 (Update 2)

### Change Summary
- Moved the frontend implementation for this flow from `UsersTab.jsx` to `AccountManagementTab.jsx`.
- Kept Account Management as a separate admin-only navigation entry.

### Files Modified
- frontend/src/components/Home.jsx
- frontend/src/components/tabs/AccountManagementTab.jsx
- frontend/src/components/tabs/UsersTab.jsx

### Impact
- The admin workflow remains unchanged, but it no longer replaces the general Users tab.
