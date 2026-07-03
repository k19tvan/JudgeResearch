"""Groq ModelClient integration for chat completions."""

import os
from typing import Optional

from api.openai_client import OpenAIClient


class GroqClient(OpenAIClient):
    """OpenAI-compatible client configured for Groq chat completions.

    Groq does not provide embeddings through this path. DeepWiki embeddings remain
    controlled by api/config/embedder.json and DEEPWIKI_EMBEDDER_TYPE.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            env_api_key_name="GROQ_API_KEY",
            env_base_url_name="GROQ_BASE_URL",
            **kwargs,
        )
