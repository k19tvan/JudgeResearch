# frontend/src/components/tabs/UsersTab.jsx

## Original Path
`frontend/src/components/tabs/UsersTab.jsx`

## Purpose
Dashboard Users tab placeholder.

## Key Behaviors
- Renders the standalone **Users** navigation tab.
- Preserves the pre-existing placeholder content for future non-admin user-facing functionality.
- Does not call account-management APIs.

## Validation Notes
- No form validation is currently performed by this placeholder tab.

## Related Files
- [src_frontend_components_Home.md](src_frontend_components_Home.md)
- [src_frontend_components_tabs_AccountManagementTab.md](src_frontend_components_tabs_AccountManagementTab.md)

# Updates

## 2026-06-02

### Change Summary
- Replaced the placeholder Users tab with the User/Account Management interface.

### Files Modified
- frontend/src/components/tabs/UsersTab.jsx
- frontend/src/api.js
- backend/main.py

### Impact
- Administrators can manage account information, roles, and active/disabled status from the dashboard.

## 2026-06-02 (Update 2)

### Change Summary
- Restored the original Users tab placeholder.
- Moved the admin-only account-management UI into a separate `AccountManagementTab.jsx` component.

### Files Modified
- frontend/src/components/tabs/UsersTab.jsx
- frontend/src/components/tabs/AccountManagementTab.jsx
- frontend/src/components/Home.jsx

### Impact
- The Users tab and Account Management tab are separate navigation entries again.
