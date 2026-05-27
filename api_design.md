# API Design Document

**Version:** 2.0  
**Last Updated:** May 26, 2026  
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Base Information](#base-information)
3. [Authentication](#authentication)
4. [User Management](#user-management)
5. [Problem Management](#problem-management)
6. [Roadmap Management](#roadmap-management)
7. [Submissions](#submissions)
8. [Error Handling](#error-handling)
9. [Response Format](#response-format)

---

## Overview

This API provides a comprehensive platform for:
- **Problem Management**: Create, retrieve, and filter coding problems
- **Roadmap Creation**: Generate learning roadmaps from GitHub repositories using AI
- **Live Coding**: Submit and track coding solutions
- **User Management**: User profiles and authentication
- **Approval Workflow**: Admin approval for public problems

### Technology Stack
- **Backend Framework**: FastAPI (Python)
- **Database**: SQLite3
- **Authentication**: JWT + Refresh Tokens
- **AI Integration**: Groq API + DeepWiki
- **File Storage**: Markdown files on disk + paths in database

---

## Base Information

### Base URLs
```
Production: http://localhost:51083
Development: http://localhost:51083
```

### API Version
- Current: v1 (implicit)
- All endpoints: `/api/*`

### Response Format
All endpoints return JSON with consistent format (see [Response Format](#response-format))

### Authentication
- **Type**: JWT Bearer Token
- **Location**: `Authorization: Bearer <access_token>` header
- **Token Expiry**: 30 minutes (auto-refresh with refresh token)
- **Refresh Token Expiry**: 7 days

---

## Authentication

### POST /api/auth/register
Register a new user account.

**Request:**
```json
{
  "username": "john_doe",
  "password": "secure_password",
  "display_name": "John Doe",
  "email": "john@example.com"
}
```

**Response (201 Created):**
```json
{
  "message": "User registered successfully"
}
```

**Error Responses:**
- `400 Bad Request`: Username or email already exists
- `500 Internal Server Error`: Database error

---

### POST /api/auth/login
Authenticate user and receive tokens.

**Request:**
```json
{
  "username": "john_doe",
  "password": "secure_password"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "token_type": "bearer"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid username or password
- `500 Internal Server Error`: Database error

**Client Storage:**
```javascript
localStorage.setItem('access_token', response.access_token);
localStorage.setItem('refresh_token', response.refresh_token);
localStorage.setItem('user_id', userId); // Extract from JWT if needed
```

---

### POST /api/auth/logout
Revoke refresh token and logout.

**Request:**
```json
{
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

**Response (200 OK):**
```json
{
  "message": "Logged out successfully"
}
```

**Client Action:**
```javascript
localStorage.removeItem('access_token');
localStorage.removeItem('refresh_token');
localStorage.removeItem('user_id');
// Redirect to login page
```

---

### POST /api/auth/refresh
Refresh expired access token using refresh token.

**Request:**
```json
{
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or expired refresh token

---

## User Management

### GET /api/users/{user_id}
Get user profile information.

**Parameters:**
- `user_id` (path): User ID (integer)

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "id": 1,
    "username": "john_doe",
    "display_name": "John Doe",
    "email": "john@example.com",
    "avatar_url": "https://example.com/avatar.jpg",
    "role": "user",
    "problems_solved": 5
  }
}
```

**Error Responses:**
- `404 Not Found`: User does not exist
- `500 Internal Server Error`: Database error

---

### PUT /api/users/{user_id}
Update user profile information.

**Parameters:**
- `user_id` (path): User ID (integer)

**Request:**
```json
{
  "display_name": "John D.",
  "email": "john.d@example.com"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Profile updated successfully"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid email format or email already exists
- `404 Not Found`: User does not exist
- `500 Internal Server Error`: Database error

---

## Problem Management

### POST /api/problems/create/manual
Create a problem manually with markdown content.

**Request:**
```json
{
  "name": "Image Classification with CNN",
  "source": "Custom",
  "statement_markdown": "# Problem Statement\n...",
  "theory_markdown": "# Theory\n...",
  "tutorial_markdown": "# Tutorial\n...",
  "solution_markdown": "# Solution\n...",
  "coding_markdown": "# Code Template\n...",
  "author_id": 1
}
```

**Fields:**
- `name` (required): Problem name
- `source` (optional): Source of the problem
- `statement_markdown` (required): Problem description
- `theory_markdown` (optional): Theoretical background
- `tutorial_markdown` (optional): Step-by-step guide
- `solution_markdown` (optional): Complete solution
- `coding_markdown` (optional): Code template
- `author_id` (required): User ID creating the problem

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Create problem manually successfully",
  "data": {
    "id": 5,
    "name": "Image Classification with CNN",
    "statement_path": "storage/problems/image_classification_with_cnn/statement.md",
    "theory_path": "storage/problems/image_classification_with_cnn/theory.md",
    "tutorial_path": "storage/problems/image_classification_with_cnn/tutorial.md",
    "solution_path": "storage/problems/image_classification_with_cnn/solution.md",
    "coding_path": "storage/problems/image_classification_with_cnn/coding.md",
    "author_id": 1,
    "is_public": 1,
    "request_status": "NONE"
  }
}
```

**Error Responses:**
- `400 Bad Request`: Missing required fields or problem name already exists
- `500 Internal Server Error`: File IO or database error

**File Storage:**
- Files saved to: `storage/problems/{name_slug}/`
- Files created: `statement.md`, `theory.md`, `tutorial.md`, `solution.md`, `coding.md`

---

### GET /api/problems/{problem_id}/content
Retrieve full problem content including all markdown files.

**Parameters:**
- `problem_id` (path): Problem ID (integer)

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "id": 5,
    "name": "Image Classification with CNN",
    "statement_markdown": "# Problem Statement\n...",
    "theory_markdown": "# Theory\n...",
    "tutorial_markdown": "# Tutorial\n...",
    "solution_markdown": "# Solution\n...",
    "coding_markdown": "# Code Template\n..."
  }
}
```

**Error Responses:**
- `404 Not Found`: Problem does not exist
- `500 Internal Server Error`: File read or database error

---

### GET /api/problems/filter
Filter problems by visibility mode.

**Query Parameters:**
- `filter_mode` (required): One of `public`, `private`, `all`
- `user_id` (required for `private` and `all` modes): User ID

**Examples:**
```
GET /api/problems/filter?filter_mode=public
GET /api/problems/filter?filter_mode=private&user_id=1
GET /api/problems/filter?filter_mode=all&user_id=1
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {
      "id": 5,
      "name": "Image Classification with CNN",
      "author_id": 1,
      "source": "Custom",
      "is_public": 1,
      "request_status": "APPROVED",
      "created_at": "2024-05-26T09:00:00"
    }
  ]
}
```

**Filter Modes:**
- `public`: Only approved public problems
- `private`: Only user's private draft problems
- `all`: All problems created by user (public + private)

**Error Responses:**
- `400 Bad Request`: Missing user_id for private/all modes
- `500 Internal Server Error`: Database error

---

### POST /api/problems/{problem_id}/request-approval
User requests admin approval to make problem public.

**Parameters:**
- `problem_id` (path): Problem ID (integer)

**Request:**
```json
{}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Approval request submitted. Admin will review shortly."
}
```

**Conditions:**
- Problem must exist
- Problem must be private (is_public = 0)
- Problem must belong to requesting user

**Error Responses:**
- `404 Not Found`: Problem does not exist
- `400 Bad Request`: Problem is not private or doesn't belong to user
- `500 Internal Server Error`: Database error

---

### POST /api/problems/{problem_id}/approve
Admin approves a problem for public release.

**Parameters:**
- `problem_id` (path): Problem ID (integer)

**Request:**
```json
{}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Problem approved and published"
}
```

**Admin Only**: Requires admin role in JWT token

**Conditions:**
- Problem must have request_status = 'PENDING'
- Problem ownership not required for admins

**Error Responses:**
- `404 Not Found`: Problem not found
- `403 Forbidden`: User not admin
- `400 Bad Request`: Problem not in PENDING status
- `500 Internal Server Error`: Database error

---

### POST /api/problems/{problem_id}/reject
Admin rejects a problem approval request.

**Parameters:**
- `problem_id` (path): Problem ID (integer)

**Request:**
```json
{}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Problem approval rejected"
}
```

**Admin Only**: Requires admin role

**Conditions:**
- Problem must have request_status = 'PENDING'

**Error Responses:**
- `404 Not Found`: Problem not found
- `403 Forbidden`: User not admin
- `400 Bad Request`: Problem not in PENDING status
- `500 Internal Server Error`: Database error

---

## Roadmap Management

### POST /api/problems/problems_from_repo
Generate problem list from GitHub repository using AI.

**Request:**
```json
{
  "roadmap_name": "Deep Learning Fundamentals",
  "repository_url": "https://github.com/example/repo",
  "level": "Intermediate",
  "user_id": 1,
  "user_note": "Focus on CNN architectures",
  "framework": "PyTorch"
}
```

**Fields:**
- `roadmap_name` (required): Name of the roadmap
- `repository_url` (required): GitHub repository URL
- `level` (required): Difficulty level (easy, medium, hard, etc.)
- `user_id` (required): User creating the roadmap
- `user_note` (optional): Additional instructions for AI
- `framework` (optional): Primary framework (PyTorch, TensorFlow, etc.)

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Generated proposed problem list successfully.",
  "data": {
    "session_id": 10,
    "repository_url": "https://github.com/example/repo",
    "roadmap_name": "Deep Learning Fundamentals",
    "proposed_problems": [
      {
        "title": "Build CNN Architecture",
        "description": "Implement a convolutional neural network...",
        "target_module": "src/models/cnn.py"
      },
      {
        "title": "Implement Data Augmentation",
        "description": "Create data augmentation pipeline...",
        "target_module": "src/data/augmentation.py"
      }
    ]
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid GitHub URL
- `502 Bad Gateway`: AI generation failed
- `500 Internal Server Error`: Database or file error

**Process:**
1. Validates GitHub URL
2. Checks DeepWiki cache
3. Sends repository content + user notes to Groq AI
4. Parses AI response to extract proposed problems
5. Saves session to database with status = 'draft'

---

### GET /api/problems/draft_sessions
List user's draft roadmap sessions.

**Query Parameters:**
- `user_id` (required): User ID

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {
      "id": 10,
      "roadmap_name": "Deep Learning Fundamentals",
      "repository_url": "https://github.com/example/repo",
      "status": "draft",
      "created_at": "2024-05-26T10:00:00"
    }
  ]
}
```

**Error Responses:**
- `500 Internal Server Error`: Database error

---

### GET /api/problems/draft_sessions/{session_id}
Get detailed information about a draft session.

**Parameters:**
- `session_id` (path): Draft session ID (integer)

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "session_id": 10,
    "roadmap_name": "Deep Learning Fundamentals",
    "repository_url": "https://github.com/example/repo",
    "user_id": 1,
    "status": "draft",
    "created_at": "2024-05-26T10:00:00",
    "updated_at": "2024-05-26T10:05:00",
    "proposed_problems": [
      {
        "title": "Build CNN Architecture",
        "description": "Implement a convolutional neural network...",
        "target_module": "src/models/cnn.py"
      }
    ]
  }
}
```

**Error Responses:**
- `404 Not Found`: Session does not exist
- `500 Internal Server Error`: Database or JSON parsing error

---

### POST /api/problems/draft_sessions/feedback
Refine draft session problems based on user feedback.

**Request:**
```json
{
  "session_id": 10,
  "feedback_text": "Add more problems on optimization techniques"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Updated draft problems with user feedback successfully.",
  "data": {
    "session_id": 10,
    "proposed_problems": [
      // Updated problems based on feedback
    ]
  }
}
```

**Process:**
1. Retrieves current problems from session
2. Sends feedback + current problems to Groq AI
3. AI generates refined problem list
4. Updates session with new problems_json

**Error Responses:**
- `404 Not Found`: Session does not exist
- `500 Internal Server Error`: Database, file, or AI error

---

### POST /api/problems/draft_sessions/finalize
Convert draft session into official roadmap and create problem records.

**Request:**
```json
{
  "session_id": 10,
  "roadmap_title": "Deep Learning Fundamentals"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Roadmap created successfully. Problems initialized as private drafts.",
  "data": {
    "roadmap_id": 3,
    "name": "Deep Learning Fundamentals",
    "repository_url": "https://github.com/example/repo",
    "problems": [
      {
        "id": 6,
        "name": "Build CNN Architecture",
        "order_index": 1,
        "status": "draft"
      }
    ]
  }
}
```

**Process:**
1. Creates new roadmap record
2. Creates problem records for each proposed problem (private, status = draft)
3. Links problems to roadmap via roadmap_problems junction table
4. Updates draft session status to 'finalized'

**Database Changes:**
- INSERT into `roadmaps` table
- INSERT into `problems` table (multiple)
- INSERT into `roadmap_problems` table (multiple)
- UPDATE `draft_problem_sessions` status

**Error Responses:**
- `404 Not Found`: Session does not exist
- `400 Bad Request`: Session already finalized
- `500 Internal Server Error`: Database transaction error

---

### GET /api/roadmaps
List user's roadmaps with statistics.

**Query Parameters:**
- `user_id` (required): User ID

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {
      "id": 3,
      "name": "Deep Learning Fundamentals",
      "repository_url": "https://github.com/example/repo",
      "level": "medium",
      "status": "draft",
      "problem_count": 5,
      "created_at": "2024-05-26T10:30:00"
    }
  ]
}
```

**Error Responses:**
- `500 Internal Server Error`: Database error

---

### GET /api/roadmaps/{roadmap_id}
Get roadmap details including all linked problems.

**Parameters:**
- `roadmap_id` (path): Roadmap ID (integer)

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "id": 3,
    "name": "Deep Learning Fundamentals",
    "repository_url": "https://github.com/example/repo",
    "level": "medium",
    "user_note": "Focus on CNN architectures",
    "framework": "PyTorch",
    "status": "draft",
    "user_id": 1,
    "created_at": "2024-05-26T10:30:00",
    "problems": [
      {
        "id": 6,
        "name": "Build CNN Architecture",
        "order_index": 1,
        "status": "draft",
        "is_public": 0,
        "request_status": "NONE"
      }
    ]
  }
}
```

**Error Responses:**
- `404 Not Found`: Roadmap does not exist
- `500 Internal Server Error`: Database error

---

## Submissions

### POST /api/submissions
Submit code solution for a problem.

**Request:**
```json
{
  "user_id": 1,
  "problem_id": 6,
  "submitted_code": "def solve():\n    # solution code\n    pass"
}
```

**Fields:**
- `user_id` (required): User ID submitting code
- `problem_id` (required): Problem ID
- `submitted_code` (required): Code content

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Submission created successfully",
  "data": {
    "submission_id": 15,
    "problem_id": 6,
    "status": "SUBMITTED"
  }
}
```

**Initial Status:** SUBMITTED (waiting for evaluation)

**Error Responses:**
- `404 Not Found`: Problem or user does not exist
- `500 Internal Server Error`: Database error

---

### GET /api/submissions/{submission_id}
Get submission details including status and results.

**Parameters:**
- `submission_id` (path): Submission ID (integer)

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "id": 15,
    "user_id": 1,
    "problem_id": 6,
    "submitted_code": "def solve():\n    # solution\n    pass",
    "status": "ACCEPTED",
    "score": 95,
    "test_results": "[{\"test\": \"test_case_1\", \"passed\": true}, ...]",
    "created_at": "2024-05-26T11:00:00"
  }
}
```

**Status Values:**
- `SUBMITTED`: Initial state, waiting for evaluation
- `ACCEPTED`: All tests passed
- `REJECTED`: Some tests failed
- `ERROR`: Compilation or runtime error

**Error Responses:**
- `404 Not Found`: Submission does not exist
- `500 Internal Server Error`: Database error

---

### GET /api/users/{user_id}/submissions
Get submission history for a user.

**Parameters:**
- `user_id` (path): User ID

**Query Parameters:**
- `problem_id` (optional): Filter by specific problem

**Examples:**
```
GET /api/users/1/submissions
GET /api/users/1/submissions?problem_id=6
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {
      "id": 15,
      "problem_id": 6,
      "status": "ACCEPTED",
      "score": 95,
      "created_at": "2024-05-26T11:00:00"
    },
    {
      "id": 14,
      "problem_id": 6,
      "status": "REJECTED",
      "score": 45,
      "created_at": "2024-05-26T10:45:00"
    }
  ]
}
```

**Error Responses:**
- `500 Internal Server Error`: Database error

---

## Error Handling

### Standard Error Response
All errors return JSON with `detail` field:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning | When Used |
|------|---------|-----------|
| 200 | OK | Successful GET, PUT, POST requests |
| 201 | Created | Resource successfully created |
| 400 | Bad Request | Invalid input, missing required fields |
| 401 | Unauthorized | Invalid credentials or expired token |
| 403 | Forbidden | Insufficient permissions (admin-only endpoints) |
| 404 | Not Found | Resource does not exist |
| 500 | Internal Server Error | Database, file IO, or unexpected errors |
| 502 | Bad Gateway | External API (AI/DeepWiki) failure |

### Common Error Scenarios

**Missing Required Fields:**
```json
{
  "detail": "validation error: field 'name' is required"
}
```

**Resource Not Found:**
```json
{
  "detail": "Problem with id 999 not found"
}
```

**Invalid Email:**
```json
{
  "detail": "Invalid email format"
}
```

**AI Generation Failed:**
```json
{
  "detail": "AI generated an invalid format. Please try again with different notes."
}
```

---

## Response Format

### Success Response
```json
{
  "status": "success",
  "message": "Optional message describing the operation",
  "data": {
    // Response data (structure varies by endpoint)
  }
}
```

### Error Response
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Authentication Response
```json
{
  "access_token": "jwt_token_here",
  "refresh_token": "refresh_token_here",
  "token_type": "bearer"
}
```

---

## Implementation Notes

### Database Storage
- **Markdown Files**: Stored on disk in `storage/problems/{name_slug}/` directories
- **File Paths**: Only paths are stored in database, not file contents
- **Problem Fields**: path fields like `statement_path`, `theory_path`, etc.

### File Management
```
storage/
└── problems/
    ├── image_classification_with_cnn/
    │   ├── statement.md
    │   ├── theory.md
    │   ├── tutorial.md
    │   ├── solution.md
    │   └── coding.md
    └── other_problem_slug/
        └── ...
```

### AI Integration
- **Service**: Groq API (fast, reliable)
- **Model**: llama-3.3-70b-versatile
- **Integration**: DeepWiki for repository analysis
- **Usage**: Problem generation from repository, refinement with feedback

### Naming Convention
- Endpoint paths: lowercase with underscores (`/api/problems/problems_from_repo`)
- JSON fields: snake_case (`problem_id`, `user_id`, `request_status`)
- Database columns: snake_case matching JSON fields

### Database Constraints
- Usernames: Unique
- Emails: Unique
- Problem names: Unique
- Refresh tokens: Unique

---

## Example Client Implementation

### JavaScript/Fetch API
```javascript
// Login
const loginResponse = await fetch('http://localhost:51083/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username, password })
});
const { access_token, refresh_token } = await loginResponse.json();
localStorage.setItem('access_token', access_token);

// Authenticated Request
const response = await fetch('http://localhost:51083/api/users/1', {
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
  }
});
```

### Python/Requests Library
```python
import requests

BASE_URL = 'http://localhost:51083'

# Login
response = requests.post(f'{BASE_URL}/api/auth/login', json={
    'username': 'john_doe',
    'password': 'password'
})
tokens = response.json()
access_token = tokens['access_token']

# Authenticated Request
headers = {'Authorization': f'Bearer {access_token}'}
user = requests.get(f'{BASE_URL}/api/users/1', headers=headers).json()
```

---

## Rate Limiting

Currently: **Not Implemented**

Future considerations:
- Rate limit: 100 requests per minute per user
- Rate limit: 1000 requests per minute per IP

---

## Versioning

**Current Version:** 1.0  
**API Path Pattern:** `/api/*`

Future versions may use `/api/v2/*` if breaking changes are needed.

---

## Security Considerations

### Token Management
- Access tokens expire in 30 minutes
- Refresh tokens valid for 7 days
- Tokens are JWT encoded (not database lookups)
- Refresh token revocation supported via logout

### Password Security
- Passwords hashed with bcrypt
- Never returned in API responses
- Only accepted via secure POST endpoints

### CORS Configuration
- Allowed origins: `*`, `http://localhost:51083`, `http://localhost:18026`
- Credentials: Allowed
- Methods: All (`*`)
- Headers: All (`*`)

### Input Validation
- All string inputs trimmed
- Email format validated
- GitHub URLs parsed and validated
- JSON parsed and validated before processing

---

## Appendix: Database Schema Quick Reference

### users
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PK |
| username | TEXT | UNIQUE, NOT NULL |
| password_hash | TEXT | NOT NULL |
| display_name | TEXT | |
| email | TEXT | UNIQUE, NOT NULL |
| avatar_url | TEXT | |
| role | TEXT | DEFAULT 'user' |
| created_at | DATETIME | |
| updated_at | DATETIME | |

### problems
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PK |
| author_id | INTEGER | FK users(id) |
| name | TEXT | UNIQUE, NOT NULL |
| statement_path | TEXT | |
| theory_path | TEXT | |
| tutorial_path | TEXT | |
| solution_path | TEXT | |
| coding_path | TEXT | |
| is_public | BOOLEAN | DEFAULT 1 |
| request_status | TEXT | DEFAULT 'NONE' |
| created_at | DATETIME | |
| updated_at | DATETIME | |

### roadmaps
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PK |
| user_id | INTEGER | FK users(id) |
| name | TEXT | NOT NULL |
| repository_url | TEXT | |
| level | TEXT | |
| user_note | TEXT | |
| framework | TEXT | |
| status | TEXT | DEFAULT 'draft' |
| created_at | DATETIME | |
| updated_at | DATETIME | |

### submissions
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PK |
| user_id | INTEGER | FK users(id) |
| problem_id | INTEGER | FK problems(id) |
| submitted_code | TEXT | |
| status | TEXT | DEFAULT 'SUBMITTED' |
| score | INTEGER | |
| test_results | TEXT | |
| created_at | DATETIME | |

---

## Related Documentation

- [API_USECASE_FLOW.md](API_USECASE_FLOW.md) - Detailed use case flows with database changes
- [ENDPOINT_CLEANUP.md](ENDPOINT_CLEANUP.md) - Endpoint removal and consolidation
- [database_design.md](database_design.md) - Complete database schema

---

**Last Updated:** May 26, 2026  
**Author:** Development Team  
**Status:** Ready for Production
