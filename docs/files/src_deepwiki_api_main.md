# deepwiki-open/api/main.py

## Original Path
`deepwiki-open/api/main.py`

## Purpose
The primary REST controller for the DeepWiki python service, orchestrating document ingestion and AI documentation extraction.

## Imports
- `fastapi`, `uvicorn`.
- internal `rag.py` logic.
- vector storage mechanisms.

## Execution Notes
Serves on Port `21082`. Works conjunctly with the Next.js visualizer frontend.

## Modification Risks
Changes here directly disrupt the Research workflow if endpoints mismatched between DeepWiki front/back.

## Related Files
- [src_deepwiki_api_rag.md](src_deepwiki_api_rag.md)
