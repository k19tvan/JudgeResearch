# Support Ticket Flow

## Overview
Describes how end-users create support tickets and attach images, and how administrators manage and resolve them.

## Step-by-Step Execution
1. **Creation**: User opens `TicketsTab.jsx` and submits a title, description, and optional image file via `multipart/form-data`.
2. **Storage**: `main.py` saves the image locally to `storage/tickets/` and records the path in SQLite (`image_url`).
3. **Display**: The UI fetches the ticket details and renders the attached image using the statically mounted `/tickets_media` route.
4. **Resolution**: Administrators can reply to the thread and toggle the status to "resolved".

## 2026-06-03 (Update 2)

### Change Summary
- Shifted ticket creation to extract `user_id` securely from the JWT `Authorization` header instead of `FormData` to prevent SQLite Foreign Key constraint violations.
- Implemented multiple image upload capabilities (`List[UploadFile]`).
- Updated the database schema implementation to store `image_url` as a JSON string array.

### Impact
- Form payloads no longer require `user_id`. Frontend correctly attaches Bearer tokens to `/api/tickets` requests.
- Users can select and view multiple screenshots per ticket.

## 2026-06-03 (Update 3)

### Change Summary
- Extended the ticket reply flow to support `multipart/form-data` requests.
- Authorized replies securely via the JWT `Authorization` header instead of raw JSON `user_id`.
- Added image upload capabilities (`List[UploadFile]`) for responses within the conversation thread.