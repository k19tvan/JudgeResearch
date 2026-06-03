# frontend/src/components/tabs/AccountManagementTab.jsx

## Original Path
`frontend/src/components/tabs/AccountManagementTab.jsx`

## Purpose
Admin-only account management UI for listing, searching, selecting, and updating user accounts.

## Key Behaviors
- Loads managed accounts from the admin API.
- Supports account search by username, display name, email, role, or status.
- Displays account summary rows with avatar, email, role, status, and last updated date.
- Opens a detail form for the selected account.
- Lets admins update display name, email, role, and status.
- Supports **Save**, **Cancel**, and **Back** actions.
- Displays **Update successful** or backend validation errors.

## Validation Notes
- Display name is required.
- Email is required and must match standard email format before submission.
- Backend validation remains authoritative for duplicate email, allowed role, allowed status, and admin authorization.

## Related Files
- [src_frontend_api.md](src_frontend_api.md)
- [src_backend_main.md](src_backend_main.md)
- [src_frontend_components_Home.md](src_frontend_components_Home.md)
- [src_frontend_components_tabs_UsersTab.md](src_frontend_components_tabs_UsersTab.md)

# Updates

## 2026-06-02

### Change Summary
- Added a dedicated admin Account Management tab component.
- Moved the account-management UI out of `UsersTab.jsx`.

### Files Modified
- frontend/src/components/tabs/AccountManagementTab.jsx
- frontend/src/components/tabs/UsersTab.jsx
- frontend/src/components/Home.jsx

### Impact
- Admin account management no longer replaces the existing Users tab.
