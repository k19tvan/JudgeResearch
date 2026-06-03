# System Architecture

## Overview
JudgeResearch is a comprehensive educational platform that integrates problem management, interactive live coding, and AI-assisted documentation visualization via DeepWiki. 

The system operates as a classic distributed web application with a standalone microservice for deep repository research.

## High-Level Components
1. **Frontend Application**: React + Vite SPA. Connects to backend APIs for user, problem, and submission management.
2. **Main Backend API**: FastAPI Python server. Manages core business logic: user auth, contests, problem statements, and integration with local evaluation sandboxes.
3. **DeepWiki Subsystem**: A dedicated AI-powered backend and frontend module deployed together. Provides "Deep Research" capabilities directly from Git repositories.
4. **Database**: SQLite database initialized natively for local environments, storing user data, problems, and submission history.

## Data Flow
- Users interact with the React Frontend (`localhost:21080`).
- Frontend communicates securely via REST an OAuth/token approach with the Main Backend (`localhost:21081`).
- The Main Backend reads/writes to the `.db` SQLite instances.
- The user can also request Repository investigations, passing parameters to DeepWiki (`localhost:21082`) which interacts with LLMs to stream AI feedback.

## Deployment Architecture
Components run locally on distinct ports:
- **21080**: React Interface
- **21081**: FastAPI Judge/Platform Endpoints
- **21082**: DeepWiki API Server
