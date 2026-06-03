# frontend/src/components/tabs/ProfileTab.jsx

## Original Path
`frontend/src/components/tabs/ProfileTab.jsx`

## Purpose
Displays the logged-in user's profile, supports profile editing, handles admin privilege activation, and provides confirmed account deactivation.

## Key Behaviors
- Loads the current user's profile with the authenticated profile API helper.
- Shows saved username, display name, email, member date, and account role.
- Provides **Edit Profile** mode for username, display name, email, and optional password updates.
- Provides **Back** behavior that restores saved profile data and exits edit mode without calling the update API.
- Runs client-side validation before update submission.
- Displays backend validation errors and the **Update successful** success message.
- Sends confirmed account deactivation to the backend, clears local session data, and returns to login.
- Syncs the dashboard avatar after profile load/update and clears avatar storage during deactivation.

## Validation Notes
- Username is required, must be 3-32 characters, and may contain only letters, numbers, dots, underscores, and dashes.
- Username cannot match the effective display name in the edit form.
- Display name and email are required.
- Email must follow standard email format.
- Password is optional. When supplied, it must be longer than 8 characters, contain at least one digit, and contain no whitespace.
- Server-side checks remain authoritative for username/display-name uniqueness, username community blocklist, email uniqueness, token ownership, and deactivation cleanup.

## Related Files
- [src_frontend_api.md](src_frontend_api.md)
- [src_backend_main.md](src_backend_main.md)
- [src_backend_auth.md](src_backend_auth.md)

# Updates

## 2026-06-01

### Change Summary
- Added the edit profile form, Back behavior, update submission, and account deactivation action.
- Added client-side validation aligned with the revised account update rules.

### Files Modified
- frontend/src/components/tabs/ProfileTab.jsx
- frontend/src/api.js

### Impact
- The Profile tab now implements Use Case 1.3 instead of only displaying profile information.

## 2026-06-02

### Change Summary
- Added avatar URL cleanup during account deactivation.

### Files Modified
- frontend/src/components/tabs/ProfileTab.jsx
- frontend/src/components/Home.jsx
- frontend/src/components/Login.jsx

### Impact
- Deleted/deactivated sessions do not leave stale avatar state for the next login.
