# JudgeResearch Platform

A comprehensive educational problem management and research platform that combines AI-powered code analysis with interactive problem creation and evaluation.

## Features

- **Research Tab**: Create learning roadmaps from GitHub repositories using AI analysis
- **Problems Tab**: Manage and organize educational problems with detailed materials
- **Detailed Problem Creation**: Generate comprehensive problem content (statement, theory, tutorial, solution, coding template)
- **Problem Approval Workflow**: Request and manage approval for public problems
- **Submissions**: Users can submit solutions and track their progress
- **User Management**: Registration, authentication, and user profile management
- **DeepWiki Integration**: Advanced repository analysis and wiki generation
- **Multi-language Support**: Support for multiple languages

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite
- **API Client Integration**: Groq, OpenAI, Google Gemini, OpenRouter, Bedrock, Azure AI, DashScope
- **Authentication**: JWT tokens

### Frontend
- **Main App**: React (Vite)
- **DeepWiki**: Next.js
- **Styling**: Tailwind CSS
- **Icons**: React Icons

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd JudgeResearch
```

### 2. Backend Setup

#### Create Python Virtual Environment

```bash
# Create virtual environment
cd backend
conda create -n env python=3.10
pip install -r requirements.txt
```

#### Initialize Database

```bash
python -m database.initialize_database
```

This will create the SQLite database with all necessary tables.

### 3. Frontend Setup

#### Install Frontend Dependencies

```bash
cd frontend
npm install
```


### 4. DeepWiki Setup (Optional)

The DeepWiki service is optional but provides advanced repository analysis features.

```bash
cd deepwiki-open
conda create -n deepwiki python=3.11
python -m pip install poetry==2.0.1 && poetry install -C api
```

**Note**: DeepWiki requires Docker and specific environment setup. See `deepwiki-open/README.md` for detailed instructions.

## Running the Application

### Start Backend Server

```bash
conda activate env
# Run the backend API
python -m uvicorn backend.main:app --host 0.0.0.0 --port 57100 --reload
```

The backend API will be available at: `http://localhost:57100`

### Start Frontend Development Server

In a new terminal:

```bash
cd frontend
npm run dev -- --port 21080 --host 0.0.0.0
```

The frontend will be available at: `http://localhost:21080` (or as shown in console)

### Start DeepWiki 

In another terminal:

```bash
cd deepwiki-open
python -m api.main
```

DeepWiki will be available at: `http://localhost:18026`

## Project Structure

```
.
├── backend/                    # FastAPI backend
│   ├── main.py                # Main application file with all endpoints
│   ├── auth.py                # Authentication utilities
│   ├── requirements.txt        # Python dependencies
│   └── ...
├── database/                   # Database utilities
│   ├── initialize_database.py # Database initialization script
│   └── migrations/            # Database migrations
├── frontend/                   # React frontend (Vite)
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── tabs/            # Tab components (Research, Problems, etc.)
│   │   ├── api.js           # API client functions
│   │   └── App.jsx          # Main app component
│   └── package.json
├── deepwiki-open/             # Next.js wiki generation service
├── storage/                   # Local storage for problems
│   └── problems/             # Problem files organized by slug
├── prompts/                   # AI prompt templates
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## API Endpoints Overview

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/logout` - Logout user
- `POST /api/auth/refresh` - Refresh access token

### Problems
- `GET /api/problems/filter` - Filter problems
- `POST /api/problems/create/manual` - Create problem manually
- `GET /api/problems/{problem_id}/content` - Get problem content
- `POST /api/problems/{problem_id}/create_detailedly` - Generate detailed content via AI
- `POST /api/problems/{problem_id}/request-approval` - Request problem approval
- `POST /api/problems/{problem_id}/approve` - Approve problem

### Research & Roadmaps
- `POST /api/problems/problems_from_repo` - Create problems from repository
- `GET /api/problems/draft_sessions` - Get draft sessions
- `GET /api/problems/draft_sessions/{session_id}` - Get session details
- `POST /api/problems/draft_sessions/feedback` - Update with feedback
- `POST /api/problems/draft_sessions/finalize` - Finalize roadmap
- `GET /api/roadmaps` - List roadmaps
- `GET /api/roadmaps/{roadmap_id}` - Get roadmap details

### Submissions
- `POST /api/submissions` - Create submission
- `GET /api/submissions/{submission_id}` - Get submission
- `GET /api/users/{user_id}/submissions` - Get user submissions

### User
- `GET /api/users/{user_id}` - Get user profile
- `PUT /api/users/{user_id}` - Update user profile

## Configuration

### Database Configuration

The database is automatically initialized with the following tables:
- `users` - User accounts
- `problems` - Problem definitions
- `submissions` - User submissions
- `roadmaps` - Learning roadmaps
- `roadmap_problems` - Problem-roadmap relationships
- `draft_problem_sessions` - Draft sessions for problem creation
- `refresh_tokens` - Session tokens

### API Provider Configuration

The backend supports multiple AI providers. Configure them in `.env`:

- **Groq** (Recommended for problem generation)
- **OpenAI** (GPT models)
- **Google Gemini**
- **OpenRouter** (Multiple models)
- **AWS Bedrock**
- **Azure AI**
- **DashScope**

## Troubleshooting

### Backend Connection Issues

If frontend can't connect to backend:
1. Ensure backend is running on port 57100
2. Check CORS settings in `backend/main.py`
3. Verify `frontend/.env` has correct `VITE_BACKEND_URL`

### Database Errors

If database errors occur:
1. Delete `database/database.db`
2. Run `python database/initialize_database.py` again
3. Restart the backend server

### Missing Dependencies

If you get import errors:
1. Ensure virtual environment is activated
2. Run `pip install -r backend/requirements.txt` again
3. For frontend: run `npm install` again

## Development

### Code Style

- Python: Follow PEP 8 guidelines
- JavaScript/React: Use ESLint configuration

### Git Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Create Pull Request

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Contact the development team

## Roadmap

- [ ] Real-time collaboration on problems
- [ ] Advanced analytics dashboard
- [ ] Integration with more AI providers
- [ ] Mobile app
- [ ] Advanced plagiarism detection
- [ ] Problem versioning system
- [ ] Community contributions

---

**Last Updated**: May 2026
