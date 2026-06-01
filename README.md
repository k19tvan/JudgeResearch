# JudgeResearch Platform

JudgeResearch is an educational problem management and research platform that combines AI-powered code analysis, automated evaluation, and interactive problem creation. The platform enables instructors and learners to create, manage, analyze, and evaluate programming problems through an integrated web interface.

## Architecture

The project consists of three main services:

| Service     | Description                                                             | Port    |
| ----------- | ----------------------------------------------------------------------- | ------- |
| Frontend    | User interface built with modern web technologies                       | `21080` |
| Backend API | Core application logic, authentication, evaluation, and database access | `21081` |
| DeepWiki    | AI-powered research and documentation service                           | `21082` |

---

## Prerequisites

Before starting, ensure the following tools are installed:

* Python 3.11+
* Node.js 18+
* npm
* Git
* Conda (optional but recommended)

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/k19tvan/JudgeResearch.git
cd JudgeResearch
```

---

## 2. Backend Setup

### Create Virtual Environment

```bash
cd backend

python -m venv .venv
.venv/Scripts/Activate.ps1

pip install -r requirements.txt
```

### Initialize Database

```bash
python -m database.initialize_database
```

This command creates the SQLite database and initializes all required tables.

---

## 3. Frontend Setup

Install frontend dependencies:

```bash
cd frontend

npm install
```

---

## 4. DeepWiki Setup

```bash
cd deepwiki-open

python -m venv .venv
.venv/Scripts/Activate.ps1

python -m pip install poetry==2.0.1

poetry install -C api

pip install dotenv grpcio faiss-cpu
```

Create environment file:

```bash
cp env_example .env
```

Then add your Gemini API key inside `.env`.

---

# Configuration

## Backend Environment

Create a backend environment file:

```bash
cp .env.example .env
```

Update important values:

```env
ADMIN_SECRET_KEY=your-secret-key
```

---

# Running the Application

## 1. Start Backend API

```bash
cd JudgeResearch

conda activate env

python -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 21081 \
    --reload
```

Backend API:

```text
http://localhost:21081
```

---

## 2. Start DeepWiki Service

Open a new terminal:

```bash
cd deepwiki-open

python -m api.main
```

DeepWiki service:

```text
http://localhost:21082
```

---

## 3. Start Frontend

Open another terminal:

```bash
cd frontend

npm run dev -- --host 0.0.0.0 --port 21080
```

Frontend application:

```text
http://localhost:21080
```

