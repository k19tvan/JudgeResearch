# frontend/src/components/tabs/TicketsTab.jsx

## Original Path
`frontend/src/components/tabs/TicketsTab.jsx`

## Purpose
Provides the UI for users to report technical issues and for admins to resolve them. Includes a thread view for replies.

## Modification Risks
Form submission uses `FormData` instead of standard JSON in order to support image uploads. Altering the fetch header to strictly force `application/json` will break the boundary formatting for image uploads.

# Updates
## 2026-06-03
### Change Summary
- Converted ticket creation flow to handle `multipart/form-data` for image attachments.
- Rendered ticket images dynamically in the detailed view.
- Added explicit error catching via `alert` to prevent silent failing.

## 2026-06-03 (Update 2)

### Change Summary
- Added conditional rendering to hide Edit/Delete buttons on the main ticket and all replies if `selectedTicket.status === "resolved"`.
- Replaced the reply input form with a "Ticket Closed" banner when resolved.