# deepwiki-open/api/rag.py

## Original Path
`deepwiki-open/api/rag.py`

## Purpose
Core AI reasoning logic for DeepWiki. Combines fetched code context with strict system prompts to output intelligent summaries.

## Imports
- `langchain` modules or raw prompt constructors.
- Embeddings clients (`azureai_client`, `openai_client`).

## Execution Notes
Executes asynchronously and heavily depends on internet-bound connections to external LLM providers unless pointed to a local Ollama instance.

## Modification Risks
Tweaking context windows here can result in token starvation or truncation errors across large files.

## Related Files
- [src_deepwiki_api_main.md](src_deepwiki_api_main.md)
