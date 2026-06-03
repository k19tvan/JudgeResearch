# Data Flow

## Overview
This document describes how data flows between different components in the JudgeResearch system.

## 1. User Authentication Flow
1. User submits credentials via the `Login.jsx` or `Register.jsx` components.
2. Frontend sends a `POST` request to `backend/auth.py` endpoints.
3. Backend creates/verifies the user in the SQLite database.
4. Backend generates a JWT token and returns it to the client.
5. Frontend stores the token (e.g., in localStorage) and includes it in the `Authorization` header for subsequent requests.

## 2. Problem Creation and Testing
1. Instructors input a problem description via the `ProblemsTab.jsx` UI.
2. The user can invoke an AI assistant to generate problem templates (utilizing `prompts.py`).
3. Payload is saved via FastAPI (`main.py`) to the storage directory (`storage/problems/`) and database.
4. The database links the problem ID with its metadata and test cases.

## 3. DeepWiki AI Research Flow
1. The user inputs a GitHub/GitLab repository URL in the Wiki Tab.
2. The Frontend proxies the request to the `deepwiki-open` service backend.
3. The DeepWiki backend clones the target repo, processes the files using embeddings (`tools/embedder.py`).
4. Data is stored in faiss/vector DB.
5. LLMs (OpenAI, local Ollama, etc.) queried for documentation context.
6. The resulting graphs and markdown are streamed back to DeepWiki frontend components (`ProcessedProjects.tsx`, `WikiTreeView.tsx`).
