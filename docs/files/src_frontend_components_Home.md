# frontend/src/components/Home.jsx

## Original Path
`frontend/src/components/Home.jsx`

## Purpose
Main authenticated dashboard shell. It renders navigation tabs, theme switching, user session display, avatar display, and the active tab content.

## Key Behaviors
- Redirects unauthenticated visitors to login.
- Reads logged-in username, role, and avatar from local storage initialized by login/profile flows.
- Filters navigation tabs by role.
- Shows general tabs such as Users and admin-only tabs such as Pending Queue and Account Management.
- Passes theme and profile-update callbacks into tab components.
- Clears avatar storage during logout.

## Related Files
- [src_frontend_api.md](src_frontend_api.md)
- [src_frontend_components_tabs_ProfileTab.md](src_frontend_components_tabs_ProfileTab.md)
- [src_frontend_components_tabs_UsersTab.md](src_frontend_components_tabs_UsersTab.md)
- [src_frontend_components_tabs_AccountManagementTab.md](src_frontend_components_tabs_AccountManagementTab.md)

# Updates

## 2026-06-02

### Change Summary
- Renamed the admin Users navigation item to Account Management.
- Restricted Account Management navigation to admin users only.

### Files Modified
- frontend/src/components/Home.jsx
- frontend/src/components/tabs/UsersTab.jsx

### Impact
- Use Case 1.4 is discoverable only to administrators from the dashboard navigation.

## 2026-06-02 (Update 2)

### Change Summary
- Restored the **Users** navigation tab.
- Added a separate admin-only **Account Management** navigation tab.

### Files Modified
- frontend/src/components/Home.jsx
- frontend/src/components/tabs/UsersTab.jsx
- frontend/src/components/tabs/AccountManagementTab.jsx

### Impact
- Admin users now see both Users and Account Management, while non-admin users still retain the Users tab.

## 2026-06-02 (Update 3)

### Change Summary
- Added avatar URL cleanup to logout.

### Files Modified
- frontend/src/components/Home.jsx
- frontend/src/components/Login.jsx

### Impact
- Logging into a different account cannot reuse the previous account's avatar from local storage.
