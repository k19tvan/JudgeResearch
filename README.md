# JudgeResearch Platform

A comprehensive educational problem management and research platform that combines AI-powered code analysis with interactive problem creation and evaluation.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/k19tvan/JudgeResearch
cd JudgeResearch
```

### 2. Backend Setup

#### Create Python Virtual Environment

```bash
# Create virtual environment
cd backend
python -m venv .venv
.venv/Scripts/Activate.ps1
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


### 4. DeepWiki Setup 

```bash
cd deepwiki-open
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install poetry==2.0.1
poetry install -C api
pip install dotenv grpcio faiss-cpu
cp ../.env_example ./.env 
```

## Running the Application

### Start Backend Server

```bash
conda activate env
# Run the backend API
python -m uvicorn backend.main:app --host 0.0.0.0 --port 21081 --reload
```

The backend API will be available at: `http://localhost:21081`

### Start DeepWiki 

In another terminal:

```bash
cd deepwiki-open
python -m api.main
```

DeepWiki will be available at: `http://localhost:21082`


### Start Frontend Development Server

In a new terminal:

```bash
cd frontend
npm run dev -- --port 21080 --host 0.0.0.0
```

The frontend will be available at: `http://localhost:21080` 


# Add admin role 
UPDATE users SET role = 'admin' WHERE id = <user_id>;