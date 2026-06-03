# DeepWiki Research Flow

## Overview
How the subsystem analyzes a git repository mapping it into a knowledge wiki.

## Step-by-Step Execution
1. **Trigger**: User inputs a generic Git link into `WikiTab.jsx`.
2. **DeepWiki Routing**: The request gets relayed to the `deepwiki-open/api/` instance running.
3. **Fetching**: The backend clones the repository directly into a temporary namespace.
4. **Data Pipeline**: Files are categorized, chunked, and embedded into local FAISS vectorDB models.
5. **RAG Orchestration**: `api/rag.py` prompts the chosen model provider with contextual codebase snipets.
6. **Streaming Results**: Structured Markdown and Mermaid.js trees are sent to the client via WebSockets (`websocket_wiki.py`).
7. **Rendering**: DeepWiki's component parses and creates an interactive folder representation for the End User.
