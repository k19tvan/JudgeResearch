# DeepWiki Module

## Purpose
An independent AI-powered repository documentation service hosted within the JudgeResearch ecosystem. It clones code repositories, segments codebase text, embeds vectors, and generates intelligent architectural markdown using LLMs.

## Responsibilities
- Manage Docker containers or a dedicated Next.js/FastAPI instance.
- Run RAG (Retrieval-Augmented Generation) pipelines over imported codebases.
- Maintain various LLM clients (OpenAI, Bedrock, Ollama, Azure, Google).
- Provide an overarching view of a project's hierarchy, rendering Mermaid graphs and textual wiki pages.

## Public Interfaces
- DeepWiki Frontend via Next.js (`src/app/page.tsx`).
- DeepWiki API via `api/main.py` and WebSockets `api/websocket_wiki.py`.

## Dependencies
- LLM Providers (OpenAI, Anthropic, Ollama, etc.).
- Embedded local FAISS vector stores.

## Dependents
- Frontend Module (`WikiTab.jsx` interacts with generated DeepWiki references).

## Internal Design
A dual-stack module:
1. A Next.js frontend found in `src/app/` generating interactive UI layouts.
2. A Python backend inside `api/` acting as the parser and RAG core. It manages caching embeddings, checking context windows, and answering prompts via `rag.py`.

## Important Files
- [src_deepwiki_api_main.md](../files/src_deepwiki_api_main.md)
- [src_deepwiki_api_rag.md](../files/src_deepwiki_api_rag.md)
- [src_deepwiki_frontend_page.md](../files/src_deepwiki_frontend_page.md)

## Common Modification Tasks
- **How to add a new Embedder**: Create a new file like `client.py` inside `api/` conforming to the interface, and connect it to `tools/embedder.py`.
- **How to update wiki prompts**: Edit `api/prompts.py` to shape how the LLMs respond to codebase files.
- **How to update the Next.js visualizer**: Alter `src/components/Markdown.tsx` or `Mermaid.tsx`.
