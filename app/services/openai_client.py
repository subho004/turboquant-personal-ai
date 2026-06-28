"""Shared OpenAI async client factory.

Centralises construction so the API key is read from ``settings`` in one
place and the client is reused across services.
"""

from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableError


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client, or fail with a safe error."""

    if not settings.openai_api_key:
        raise ServiceUnavailableError("OpenAI API key is not configured")
    return AsyncOpenAI(api_key=settings.openai_api_key)
