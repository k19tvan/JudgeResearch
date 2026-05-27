# API & Database Flow by Use Case

Tài liệu này mô tả chi tiết cách frontend gọi API endpoints và database được cập nhật khi user tương tác với từng tính năng.

---

## 📋 Mục lục Use Cases

1. [Authentication](#1-authentication)
2. [User Profile Management](#2-user-profile-management)
3. [Manual Problem Creation](#3-manual-problem-creation)
4. [Create Roadmap from Repository](#4-create-roadmap-from-repository)
5. [Manage Roadmap & Problems](#5-manage-roadmap--problems)
6. [Live Coding Submission](#6-live-coding-submission)
7. [Problem Approval Workflow](#7-problem-approval-workflow)

---

## 1. Authentication

### Usecase: User Registration & Login

#### Flow Diagram
```
User Interface
    ↓
Frontend (React)
    ↓
API Endpoint
    ↓
Database
```

### 1.1 User Registration
**When:** User submits registration form on login page

**Frontend Action:**
```javascript
// frontend/src/components/LoginPage.jsx
const response = await fetch('http://localhost:51083/api/auth/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'john_doe',
    password: 'password123',
    display_name: 'John Doe',
    email: 'john@example.com'
  })
});
```

**Backend Processing:**
```
Endpoint: POST /api/auth/register
├─ Validate input (username unique, valid email)
├─ Hash password using bcrypt
└─ INSERT INTO users (username, password_hash, display_name, email)
    └─ Database: users table
       ├─ id: auto-generated UUID
       ├─ username: 'john_doe'
       ├─ password_hash: 'hashed...'
       ├─ display_name: 'John Doe'
       ├─ email: 'john@example.com'
       ├─ role: 'user' (default)
       ├─ created_at: current timestamp
       └─ updated_at: current timestamp
```

**Response:**
```json
{
  "message": "User registered successfully"
}
```

---

### 1.2 User Login
**When:** User enters credentials and clicks login

**Frontend Action:**
```javascript
// frontend/src/components/LoginPage.jsx
const response = await fetch('http://localhost:51083/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'john_doe',
    password: 'password123'
  })
});

// Store tokens
const data = await response.json();
localStorage.setItem('access_token', data.access_token);
localStorage.setItem('refresh_token', data.refresh_token);
localStorage.setItem('user_id', userId); // Extract from token if needed
```

**Backend Processing:**
```
Endpoint: POST /api/auth/login
├─ Find user by username in users table
├─ Verify password hash
├─ Generate JWT access token (30 min expiry)
├─ Generate refresh token (random hex)
└─ INSERT INTO refresh_tokens (user_id, token, expires_at)
    └─ Database: refresh_tokens table
       ├─ id: auto-generated
       ├─ user_id: from users table
       ├─ token: 'random_hex...'
       ├─ expires_at: current_date + 7 days
       └─ is_revoked: false (default)
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "a1b2c3d4e5f6...",
  "token_type": "bearer"
}
```

---

### 1.3 User Logout
**When:** User clicks logout button

**Frontend Action:**
```javascript
// Any component with logout button
const response = await fetch('http://localhost:51083/api/auth/logout', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    refresh_token: localStorage.getItem('refresh_token')
  })
});

// Clear local storage
localStorage.removeItem('access_token');
localStorage.removeItem('refresh_token');
localStorage.removeItem('user_id');
// Redirect to login page
```

**Backend Processing:**
```
Endpoint: POST /api/auth/logout
├─ Find refresh token in database
└─ DELETE FROM refresh_tokens WHERE token = ?
    └─ Database: refresh_tokens table
       └─ Token is removed, making it invalid
```

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

---

### 1.4 Token Refresh
**When:** Access token expires (automatically triggered)

**Frontend Action:**
```javascript
// Interceptor for API calls (frontend/src/api.js)
if (response.status === 401) {
  const refreshResult = await fetch('http://localhost:51083/api/auth/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      refresh_token: localStorage.getItem('refresh_token')
    })
  });
  
  const newData = await refreshResult.json();
  localStorage.setItem('access_token', newData.access_token);
}
```

**Backend Processing:**
```
Endpoint: POST /api/auth/refresh
├─ Verify refresh_token exists and not expired
├─ Find associated user
├─ Generate new JWT access token
└─ Database: refresh_tokens table
   └─ No changes (same token still valid)
```

---

## 2. User Profile Management

### Usecase: View & Update User Profile

### 2.1 View User Profile
**When:** User clicks on profile icon or visits profile page

**Frontend Action:**
```javascript
// frontend/src/components/ProfilePage.jsx
const userId = localStorage.getItem('user_id');
const response = await fetch(`http://localhost:51083/api/users/${userId}`, {
  method: 'GET',
  headers: {
    'Content-Type': 'application/json'
  }
});
const data = await response.json();
// Display: username, display_name, email, avatar_url, problems_solved
```

**Backend Processing:**
```
Endpoint: GET /api/users/{user_id}
├─ Find user in users table
├─ COUNT problems solved (submissions with status='ACCEPTED')
└─ Return user data with stats
    └─ Database queries:
       ├─ SELECT * FROM users WHERE id = ?
       └─ SELECT COUNT(*) FROM submissions 
          WHERE user_id = ? AND status = 'ACCEPTED'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": 1,
    "username": "john_doe",
    "display_name": "John Doe",
    "email": "john@example.com",
    "avatar_url": "https://...",
    "role": "user",
    "problems_solved": 5
  }
}
```

---

### 2.2 Update User Profile
**When:** User modifies display name or email and clicks save

**Frontend Action:**
```javascript
// frontend/src/components/ProfilePage.jsx
const userId = localStorage.getItem('user_id');
const response = await fetch(`http://localhost:51083/api/users/${userId}`, {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    display_name: 'John D.',
    email: 'john.d@example.com'
  })
});
```

**Backend Processing:**
```
Endpoint: PUT /api/users/{user_id}
├─ Validate email format and uniqueness
└─ UPDATE users SET display_name = ?, email = ? WHERE id = ?
    └─ Database: users table
       ├─ display_name: updated value
       ├─ email: updated value
       └─ updated_at: current timestamp
```

**Response:**
```json
{
  "status": "success",
  "message": "Profile updated successfully"
}
```

---

## 3. Manual Problem Creation

### Usecase: User Creates a Problem Manually

**When:** User clicks "Create Manual Problem" → Fills form → Submits

#### Step-by-Step Flow

**Frontend Action (ProblemsTab.jsx):**
```javascript
// 1. User fills form
const formData = {
  name: 'Image Classification with CNN',
  source: 'Custom',
  statement_markdown: '# Problem Statement\n...',
  theory_markdown: '# Theory\n...',
  tutorial_markdown: '# Tutorial\n...',
  solution_markdown: '# Solution\n...',
  coding_markdown: '# Code Template\n...',
  author_id: 1  // Current user
};

// 2. Submit form
const response = await fetch('http://localhost:51083/api/problems/create/manual', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(formData)
});

// 3. On success, show confirmation and reload problems list
```

**Backend Processing:**
```
Endpoint: POST /api/problems/create/manual
├─ Validate all required fields
├─ Check if problem name already exists
├─ Create folder: storage/problems/{name_slug}/
├─ Write markdown files to disk:
│  ├─ storage/problems/{name_slug}/statement.md
│  ├─ storage/problems/{name_slug}/theory.md (if provided)
│  ├─ storage/problems/{name_slug}/tutorial.md (if provided)
│  ├─ storage/problems/{name_slug}/solution.md (if provided)
│  └─ storage/problems/{name_slug}/coding.md (if provided)
│
└─ INSERT INTO problems table:
    └─ Database: problems table
       ├─ id: auto-generated
       ├─ name: 'Image Classification with CNN'
       ├─ source: 'Custom'
       ├─ statement_path: 'storage/problems/image_classification_with_cnn/statement.md'
       ├─ theory_path: 'storage/problems/image_classification_with_cnn/theory.md'
       ├─ tutorial_path: 'storage/problems/image_classification_with_cnn/tutorial.md'
       ├─ solution_path: 'storage/problems/image_classification_with_cnn/solution.md'
       ├─ coding_path: 'storage/problems/image_classification_with_cnn/coding.md'
       ├─ author_id: 1
       ├─ is_public: 1 (default, but private until approved)
       ├─ request_status: 'NONE'
       ├─ created_at: current timestamp
       └─ updated_at: current timestamp
```

**Response:**
```json
{
  "status": "success",
  "message": "Create problem manually successfully",
  "data": {
    "id": 5,
    "name": "Image Classification with CNN",
    "statement_path": "storage/problems/image_classification_with_cnn/statement.md",
    "theory_path": "...",
    "tutorial_path": "...",
    "solution_path": "...",
    "coding_path": "...",
    "author_id": 1,
    "is_public": 1,
    "request_status": "NONE"
  }
}
```

---

## 4. Create Roadmap from Repository

### Usecase: User Creates Roadmap from GitHub Repository

**Timeline:** Multi-step process with iterative refinement

### 4.1 Initial Problem Generation from Repository

**When:** User fills form and clicks "Create Proposed List"

**Frontend Action (ResearchTab.jsx):**
```javascript
// 1. User fills roadmap creation form
const formData = {
  roadmap_name: 'Deep Learning Fundamentals',
  repository_url: 'https://github.com/example/repo',
  level: 'Intermediate',
  user_id: 1,
  user_note: 'Focus on CNN architectures',
  framework: 'PyTorch'
};

// 2. Submit form
const response = await fetch('http://localhost:51083/api/problems/problems_from_repo', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(formData)
});

// 3. Response contains session_id and proposed problems
const data = await response.json();
const sessionId = data.data.session_id;
// Navigate to draft detail page: /research/draft/${sessionId}
```

**Backend Processing:**
```
Endpoint: POST /api/problems/problems_from_repo
├─ Extract owner/repo from GitHub URL
├─ Check if repository is cached in DeepWiki
├─ Call DeepWiki API to get repository content
├─ Send prompt to Groq AI with:
│  ├─ Repository content
│  ├─ Level: 'Intermediate'
│  ├─ Framework: 'PyTorch'
│  └─ User note: 'Focus on CNN architectures'
│
├─ Parse AI response to get proposed problems list
│  └─ Each problem has: title, description, target_module
│
└─ INSERT INTO draft_problem_sessions:
    └─ Database: draft_problem_sessions table
       ├─ id: auto-generated
       ├─ roadmap_name: 'Deep Learning Fundamentals'
       ├─ repository_url: 'https://github.com/example/repo'
       ├─ user_id: 1
       ├─ problems_json: '[{"title":"...", "description":"...", ...}]'
       ├─ status: 'draft'
       ├─ created_at: current timestamp
       └─ updated_at: current timestamp
```

**Response:**
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

---

### 4.2 View Draft Session & Refine Problems

**When:** User opens draft detail page and provides feedback

**Frontend Action (ResearchRoadmapPage.jsx):**
```javascript
// 1. Load draft session details
const sessionId = 10;
const response = await fetch(
  `http://localhost:51083/api/problems/draft_sessions/${sessionId}`
);
const data = await response.json();
// Display proposed problems for review/feedback

// 2. User provides feedback
const feedback = "Add more problems on optimization techniques";

// 3. Submit feedback
const feedbackResponse = await fetch(
  'http://localhost:51083/api/problems/draft_sessions/feedback',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: 10,
      feedback_text: feedback
    })
  }
);
```

**Backend Processing:**

#### 4.2a GET Draft Session Details
```
Endpoint: GET /api/problems/draft_sessions/{session_id}
├─ Query draft_problem_sessions by id
├─ Parse stored JSON to extract proposed problems
└─ Return full session data
    └─ Database query:
       SELECT * FROM draft_problem_sessions WHERE id = ?
```

**Response:**
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
    "proposed_problems": [...]
  }
}
```

#### 4.2b Update Draft with User Feedback
```
Endpoint: POST /api/problems/draft_sessions/feedback
├─ Get current session
├─ Build feedback prompt for Groq AI
├─ Send to AI with:
│  ├─ Current problems list
│  ├─ Repository content
│  └─ User feedback
│
├─ Parse AI response with refined problems
└─ UPDATE draft_problem_sessions SET problems_json = ?, updated_at = ?
    └─ Database: draft_problem_sessions table
       ├─ problems_json: updated JSON with refined problems
       └─ updated_at: current timestamp
```

**Response:**
```json
{
  "status": "success",
  "message": "Updated draft problems with user feedback successfully.",
  "data": {
    "session_id": 10,
    "proposed_problems": [
      // Updated problems including new ones based on feedback
    ]
  }
}
```

---

### 4.3 Finalize Draft Session to Create Roadmap

**When:** User reviews and clicks "Create Roadmap"

**Frontend Action:**
```javascript
// User clicks finalize button
const response = await fetch(
  'http://localhost:51083/api/problems/draft_sessions/finalize',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: 10,
      roadmap_title: 'Deep Learning Fundamentals' // Can be different from roadmap_name
    })
  }
);

// On success, navigate to roadmap detail page
```

**Backend Processing:**
```
Endpoint: POST /api/problems/draft_sessions/finalize
├─ Get draft session by id
├─ Create new roadmap record
│  └─ INSERT INTO roadmaps:
│     ├─ user_id: from session
│     ├─ name: from payload (roadmap_title)
│     ├─ repository_url: from session
│     ├─ level: 'medium' (default)
│     ├─ status: 'draft'
│     └─ Database: roadmaps table
│
├─ For each proposed problem in session:
│  ├─ Extract target_module from problem
│  ├─ Create private problem record
│  │  └─ INSERT INTO problems:
│  │     ├─ name: problem title
│  │     ├─ author_id: user_id
│  │     ├─ is_public: 0 (private)
│  │     ├─ request_status: 'NONE'
│  │     └─ paths stored as NULL (not yet created)
│  │
│  └─ Link problem to roadmap
│     └─ INSERT INTO roadmap_problems:
│        ├─ roadmap_id: new roadmap id
│        ├─ problem_id: new problem id
│        ├─ name: problem title
│        ├─ description: problem description
│        ├─ order_index: sequential number
│        └─ status: 'draft'
│
├─ Update draft session status
│  └─ UPDATE draft_problem_sessions SET status = 'finalized'
│
└─ Database tables affected:
   ├─ roadmaps: INSERT
   ├─ problems: INSERT (multiple)
   └─ roadmap_problems: INSERT (multiple)
```

**Response:**
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
      },
      // ... more problems
    ]
  }
}
```

---

## 5. Manage Roadmap & Problems

### Usecase: User Views Roadmap Details and Creates Problem Materials

### 5.1 List User's Roadmaps

**When:** User opens Research tab or visits roadmaps page

**Frontend Action:**
```javascript
// ResearchTab.jsx or RoadmapList.jsx
const userId = 1;
const response = await fetch(
  `http://localhost:51083/api/roadmaps?user_id=${userId}`
);
const data = await response.json();
// Display roadmaps list with statistics
```

**Backend Processing:**
```
Endpoint: GET /api/roadmaps?user_id={user_id}
├─ Query roadmaps by user_id
├─ For each roadmap, count problems
└─ Return roadmaps list with stats
    └─ Database queries:
       ├─ SELECT * FROM roadmaps WHERE user_id = ?
       └─ For each roadmap: SELECT COUNT(*) FROM roadmap_problems
```

**Response:**
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

---

### 5.2 View Roadmap Details

**When:** User clicks on a roadmap to view problems

**Frontend Action:**
```javascript
// ResearchRoadmapPage.jsx
const roadmapId = 3;
const response = await fetch(
  `http://localhost:51083/api/roadmaps/${roadmapId}`
);
const data = await response.json();
// Display roadmap with all problems and their details
```

**Backend Processing:**
```
Endpoint: GET /api/roadmaps/{roadmap_id}
├─ Get roadmap record
├─ Get all roadmap_problems linked to roadmap
├─ For each problem, get problem details
└─ Return complete roadmap with problems hierarchy
    └─ Database queries:
       ├─ SELECT * FROM roadmaps WHERE id = ?
       ├─ SELECT * FROM roadmap_problems WHERE roadmap_id = ?
       └─ For each: SELECT * FROM problems WHERE id = ?
```

**Response:**
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
      },
      // ... more problems
    ]
  }
}
```

---

### 5.3 View Problem Content

**When:** User clicks on a problem in roadmap to see full content

**Frontend Action:**
```javascript
// ResearchRoadmapPage.jsx or ProblemDetailPage.jsx
const problemId = 6;
const response = await fetch(
  `http://localhost:51083/api/problems/${problemId}/content`
);
const data = await response.json();
// Display all markdown: statement, theory, tutorial, solution, coding
```

**Backend Processing:**
```
Endpoint: GET /api/problems/{problem_id}/content
├─ Get problem record to find file paths
├─ Read each markdown file from disk:
│  ├─ Read statement.md
│  ├─ Read theory.md (if exists)
│  ├─ Read tutorial.md (if exists)
│  ├─ Read solution.md (if exists)
│  └─ Read coding.md (if exists)
│
└─ Return all content combined
    └─ Database queries:
       SELECT * FROM problems WHERE id = ?
       (Then read files from paths stored in DB)
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": 6,
    "name": "Build CNN Architecture",
    "statement_markdown": "# Problem Statement\n...",
    "theory_markdown": "# Theory\n...",
    "tutorial_markdown": "# Tutorial\n...",
    "solution_markdown": "# Solution\n...",
    "coding_markdown": "# Code Template\n..."
  }
}
```

---

## 6. Live Coding Submission

### Usecase: User Submits Code Solution

**When:** User writes code and clicks "Submit"

**Frontend Action (ProblemDetailPage.jsx):**
```javascript
// 1. Get problem ID and user's code
const userId = 1;
const problemId = 6;
const submittedCode = `
def build_cnn():
    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        # ... more layers
    ])
    return model
`;

// 2. Submit code
const response = await fetch('http://localhost:51083/api/submissions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: userId,
    problem_id: problemId,
    submitted_code: submittedCode
  })
});

// 3. Get submission ID from response
const data = await response.json();
const submissionId = data.data.submission_id;
```

**Backend Processing:**
```
Endpoint: POST /api/submissions
├─ Validate problem exists
├─ Validate user exists
├─ Save submission
└─ INSERT INTO submissions:
    └─ Database: submissions table
       ├─ id: auto-generated
       ├─ user_id: from payload
       ├─ problem_id: from payload
       ├─ submitted_code: code content
       ├─ status: 'SUBMITTED' (initial status)
       ├─ score: NULL (to be set after grading)
       ├─ test_results: NULL (to be set after testing)
       └─ created_at: current timestamp
```

**Response:**
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

---

### 6.1 View Submission Status

**When:** User wants to check their submission status/score

**Frontend Action:**
```javascript
const submissionId = 15;
const response = await fetch(
  `http://localhost:51083/api/submissions/${submissionId}`
);
const data = await response.json();
// Display: status, score, test results
```

**Backend Processing:**
```
Endpoint: GET /api/submissions/{submission_id}
├─ Query submission by id
└─ Return full submission data
    └─ Database query:
       SELECT * FROM submissions WHERE id = ?
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": 15,
    "user_id": 1,
    "problem_id": 6,
    "submitted_code": "...",
    "status": "ACCEPTED",
    "score": 95,
    "test_results": "[{\"test\": \"test_case_1\", \"passed\": true}, ...]",
    "created_at": "2024-05-26T11:00:00"
  }
}
```

---

### 6.2 View User's Submission History

**When:** User views their submissions for a problem or all problems

**Frontend Action:**
```javascript
// View submissions for specific problem
const userId = 1;
const problemId = 6;
const response = await fetch(
  `http://localhost:51083/api/users/${userId}/submissions?problem_id=${problemId}`
);

// OR view all submissions
const allResponse = await fetch(
  `http://localhost:51083/api/users/${userId}/submissions`
);
```

**Backend Processing:**
```
Endpoint: GET /api/users/{user_id}/submissions?problem_id={problem_id}
├─ If problem_id provided:
│  └─ SELECT * FROM submissions 
│     WHERE user_id = ? AND problem_id = ?
│
└─ Otherwise:
   └─ SELECT * FROM submissions WHERE user_id = ?
       └─ Database: submissions table
```

**Response:**
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

---

## 7. Problem Approval Workflow

### Usecase: User Requests Problem Approval, Admin Approves/Rejects

### 7.1 Request Problem Approval

**When:** User clicks "Request Public" button on their problem

**Frontend Action:**
```javascript
// ProblemDetailPage.jsx or ProblemsTab.jsx
const problemId = 5;
const response = await fetch(
  `http://localhost:51083/api/problems/${problemId}/request-approval`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  }
);
```

**Backend Processing:**
```
Endpoint: POST /api/problems/{problem_id}/request-approval
├─ Verify problem exists
├─ Verify problem is private (is_public = 0)
├─ Verify problem belongs to current user
│
└─ UPDATE problems SET request_status = 'PENDING'
    └─ Database: problems table
       ├─ request_status: 'PENDING' (was 'NONE')
       └─ updated_at: current timestamp
```

**Response:**
```json
{
  "status": "success",
  "message": "Approval request submitted. Admin will review shortly."
}
```

---

### 7.2 Filter Problems by Visibility Mode

**When:** User wants to see only public problems, only their private problems, or all

**Frontend Action:**
```javascript
// ProblemsTab.jsx or ProblemsFilterPage.jsx
const userId = 1;

// View public problems only
const publicResponse = await fetch(
  `http://localhost:51083/api/problems/filter?filter_mode=public`
);

// View user's private problems only
const privateResponse = await fetch(
  `http://localhost:51083/api/problems/filter?filter_mode=private&user_id=${userId}`
);

// View all (requires user_id)
const allResponse = await fetch(
  `http://localhost:51083/api/problems/filter?filter_mode=all&user_id=${userId}`
);
```

**Backend Processing:**
```
Endpoint: GET /api/problems/filter?filter_mode={mode}&user_id={user_id}

Filter Modes:

1. filter_mode=public:
   └─ SELECT * FROM problems 
      WHERE is_public = 1 AND request_status IN ('APPROVED', 'NONE')

2. filter_mode=private (requires user_id):
   └─ SELECT * FROM problems 
      WHERE is_public = 0 AND author_id = ?

3. filter_mode=all (requires user_id):
   └─ SELECT * FROM problems 
      WHERE author_id = ?
       └─ Database: problems table
          └─ Returns all problems matching filter criteria
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": 5,
      "name": "Image Classification with CNN",
      "author_id": 1,
      "is_public": 1,
      "request_status": "APPROVED",
      "created_at": "2024-05-26T09:00:00"
    }
  ]
}
```

---

### 7.3 Admin: Approve Problem

**When:** Admin views pending approvals and clicks "Approve"

**Frontend Action (Admin Panel):**
```javascript
// AdminPanel.jsx
const problemId = 5;
const response = await fetch(
  `http://localhost:51083/api/problems/${problemId}/approve`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  }
);
```

**Backend Processing:**
```
Endpoint: POST /api/problems/{problem_id}/approve
├─ Verify admin role (check token)
├─ Find problem with request_status = 'PENDING'
│
└─ UPDATE problems SET:
    ├─ is_public = 1 (make public)
    ├─ request_status = 'APPROVED'
    └─ updated_at = current timestamp
       └─ Database: problems table
          └─ Problem is now visible to all users
```

**Response:**
```json
{
  "status": "success",
  "message": "Problem approved and published"
}
```

---

### 7.4 Admin: Reject Problem

**When:** Admin reviews problem and rejects it

**Frontend Action (Admin Panel):**
```javascript
const problemId = 5;
const response = await fetch(
  `http://localhost:51083/api/problems/${problemId}/reject`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  }
);
```

**Backend Processing:**
```
Endpoint: POST /api/problems/{problem_id}/reject
├─ Verify admin role
├─ Find problem with request_status = 'PENDING'
│
└─ UPDATE problems SET:
    ├─ is_public = 0 (keep private)
    ├─ request_status = 'REJECTED'
    └─ updated_at = current timestamp
       └─ Database: problems table
          └─ Problem stays private, status updated
```

**Response:**
```json
{
  "status": "success",
  "message": "Problem approval rejected"
}
```

---

## 📊 Database Changes Summary

### Operations by Table

| Table | Operations | Use Cases |
|-------|-----------|-----------|
| `users` | INSERT (register), SELECT (login, profile), UPDATE (profile) | Auth, User Profiles |
| `refresh_tokens` | INSERT (login), DELETE (logout) | Auth |
| `problems` | INSERT (manual create, finalize roadmap), SELECT (view, filter), UPDATE (request approval, approve, reject) | All |
| `draft_problem_sessions` | INSERT (create roadmap), SELECT (view draft), UPDATE (feedback, finalize) | Create Roadmap |
| `roadmaps` | INSERT (finalize), SELECT (list, detail) | Manage Roadmap |
| `roadmap_problems` | INSERT (finalize), SELECT (view roadmap) | Manage Roadmap |
| `submissions` | INSERT (submit code), SELECT (view status, history) | Live Coding |

---

## 🔄 API Call Order by Scenario

### Scenario 1: Register → Login → View Profile
```
1. POST /api/auth/register
2. POST /api/auth/login
3. GET /api/users/{user_id}
```

### Scenario 2: Create Manual Problem → Request Approval → (Admin) Approve
```
1. POST /api/problems/create/manual
2. POST /api/problems/{problem_id}/request-approval
3. [Admin] POST /api/problems/{problem_id}/approve
4. GET /api/problems/filter?filter_mode=public
```

### Scenario 3: Full Roadmap Creation Workflow
```
1. POST /api/problems/problems_from_repo
2. GET /api/problems/draft_sessions/{session_id}
3. POST /api/problems/draft_sessions/feedback (optional, repeat for refinement)
4. POST /api/problems/draft_sessions/finalize
5. GET /api/roadmaps/{roadmap_id}
6. GET /api/problems/{problem_id}/content (repeat for each problem)
7. POST /api/submissions (user submits code for a problem)
```

### Scenario 4: Submit Code → View History
```
1. POST /api/submissions
2. GET /api/submissions/{submission_id}
3. GET /api/users/{user_id}/submissions
```

---

## ⚠️ Important Notes

1. **File Storage**: Problem markdown files are stored on disk, only paths are in database
2. **AI Integration**: Roadmap creation uses DeepWiki + Groq AI for problem generation
3. **Approval Flow**: Only admin can approve/reject; problem must be in PENDING status
4. **User Isolation**: Users only see their own private problems and public problems
5. **Token Expiry**: Access tokens expire in 30 minutes; refresh tokens valid for 7 days

