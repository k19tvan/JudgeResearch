# Problem Creation Flow

## Overview
How an Instructor or User interacts with the platform to generate coding challenges.

## Step-by-Step Execution
1. **Initiation**: Instructor opens `ProblemsTab.jsx`.
2. **Drafting Content**: Fills out the markdown problem description and logic.
3. **AI Generation (Optional)**: Instructor invokes templating. `backend` pings `prompts.py` to draft structure via an LLM.
4. **Saving**: Payload forms a JSON encompassing (Description, Time Limit, Memory Limit, Test Cases).
5. **Backend Processing**: `backend/main.py` saves metadata to SQLite and structured text resources to `storage/problems/{problem_id}`.
6. **Publication**: Problem becomes visible via `GET /problems` on the frontend for students.
