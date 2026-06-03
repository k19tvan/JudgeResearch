# Frontend Module

## Purpose
Provides the User Interface for instructors, students, and administrators to interact with the JudgeResearch platform. It handles authentication, live coding, contest display, and wiki visualizations.

## Responsibilities
- Rendering UI components (Dashboards, Login, LiveCode environments).
- State and session management for the user.
- Communicating with the Main Backend API to fetch problems, submissions, and contest info.
- Providing profile editing controls, account update feedback, and account deactivation confirmation.
- Providing admin-only account management for user search, details, role changes, and account activation/disablement.
- Communicating with DeepWiki components to show repository mappings.

## Public Interfaces
- Web browser entry point at `index.html`.
- REST API bindings in `src/api.js`.

## Dependencies
- React and Vite for compilation.
- Tailwind CSS for styling.
- `package.json` outlines third-party libs (like monaco-editor for live coding).

## Dependents
- The End-User.

## Internal Design
The application is structured into high-level pages (`Home.jsx`, `LiveCodingPage.jsx`, `ResearchRoadmapPage.jsx`, `Login.jsx`, `Register.jsx`) and specialized tab constituents. The tabs located in `src/components/tabs/` include:
- `ProblemsTab.jsx`, `ContestsTab.jsx`, `SubmissionsTab.jsx`, `UsersTab.jsx`, `WikiTab.jsx`
- Management/Personal tabs: `AccountManagementTab.jsx`, `AdminQueueTab.jsx`, `MyRequestsTab.jsx`, `ProfileTab.jsx`, `ResearchTab.jsx`

## Important Files
- [src_frontend_App.md](../files/src_frontend_App.md)
- [src_frontend_api.md](../files/src_frontend_api.md)
- [src_frontend_components_Login.md](../files/src_frontend_components_Login.md)
- [src_frontend_components_Home.md](../files/src_frontend_components_Home.md)
- [src_frontend_components_tabs_ProfileTab.md](../files/src_frontend_components_tabs_ProfileTab.md)
- [src_frontend_components_tabs_UsersTab.md](../files/src_frontend_components_tabs_UsersTab.md)
- [src_frontend_components_tabs_AccountManagementTab.md](../files/src_frontend_components_tabs_AccountManagementTab.md)

## Common Modification Tasks
- **How to add a new tab**: Create a new component in `src/components/tabs/` and link it in the dashboard view.
- **How to edit API endpoints**: Modify `src/api.js` to add the new server routes.

# Updates

## 2026-06-01

### Change Summary
- Documented profile editing and account deactivation as frontend responsibilities.

### Files Modified
- frontend/src/api.js
- frontend/src/components/tabs/ProfileTab.jsx

### Impact
- The Profile tab now covers the complete Account Update use case instead of only displaying profile data.

## 2026-06-02

### Change Summary
- Added frontend responsibility for the admin-only Account Management tab.

### Files Modified
- frontend/src/components/Home.jsx
- frontend/src/components/tabs/UsersTab.jsx
- frontend/src/api.js

### Impact
- Administrators can open Account Management from navigation and update user account fields from the dashboard.

## 2026-06-02 (Update 2)

### Change Summary
- Documented `AccountManagementTab.jsx` as the dedicated admin account-management component.
- Restored `UsersTab.jsx` as a separate dashboard tab.

### Files Modified
- frontend/src/components/Home.jsx
- frontend/src/components/tabs/UsersTab.jsx
- frontend/src/components/tabs/AccountManagementTab.jsx

### Impact
- Frontend tab ownership is clearer: Users and Account Management are separate components.

## 2026-06-02 (Update 3)

### Change Summary
- Documented login session avatar initialization.
- Added Login file documentation to the frontend module index.

### Files Modified
- frontend/src/components/Login.jsx
- frontend/src/components/Home.jsx
- frontend/src/components/tabs/ProfileTab.jsx
- backend/main.py

### Impact
- Frontend session state now includes the correct account avatar immediately after login.
