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

## 2026-06-03 (Update 4)

### Change Summary
- Added full CRUD (Edit/Delete) functionality for Tickets and Ticket Replies.
- Allowed users to remove specific old images and append new images during edits via `kept_images` JSON arrays.
- Restricted the "Mark as Resolved" and "Re-open" status toggle actions to `admin` roles only.

### Impact
- Standardizes community management expectations. Users have full control over their content, while administrators retain exclusive authority over ticket lifecycle closures.

## 2026-06-03 (Update 5)

### Change Summary
- Implemented "Ticket Freeze" logic. When a ticket's status is set to `resolved`, it becomes completely immutable.
- Prevented end-users and admins from adding replies, updating content, or deleting the ticket/replies while in a resolved state.

### Impact
- Forces administrators to explicitly reopen a ticket (`status = open`) before any further destructive actions or communications can take place.

## 2026-06-03 (Update 6)

### Change Summary
- Transformed Support Tickets from a private helpdesk into a public community issue tracker.
- All authenticated users can now view all tickets and their associated conversation threads.
- All authenticated users can post replies to any open ticket to assist others.
- Edit and Delete operations remain strictly restricted to the original author or an administrator.