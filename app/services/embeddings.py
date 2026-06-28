"""Embedding provider.

Defines an `EmbeddingProvider` protocol so services depend on an
abstraction (tests inject a fake). The OpenAI implementation batches
requests and caches by content hash to avoid re-embedding identical text.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from app.core.config import settings
from app.core.logging import get_logger
from app.services.openai_client import get_openai_client
from app.services.usage import get_usage_meter
from utils.hashing import sha256_text

logger = get_logger(__name__)

_BATCH_SIZE = 100


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    """OpenAI embeddings with batching and an in-memory content-hash cache."""

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(sha256_text(text))
            if cached is not None:
                results[i] = cached
            else:
                pending.append((i, text))

        client = get_openai_client()
        for start in range(0, len(pending), _BATCH_SIZE):
            batch = pending[start : start + _BATCH_SIZE]
            began = time.perf_counter()
            response = await client.embeddings.create(
                model=settings.embed_model,
                input=[text for _, text in batch],
            )
            logger.info(
                "Embedded %d texts in %.3fs", len(batch), time.perf_counter() - began
            )
            if response.usage:
                get_usage_meter().add_embeddings(response.usage.total_tokens)
            for (idx, text), item in zip(batch, response.data):
                vector = list(item.embedding)
                results[idx] = vector
                self._cache[sha256_text(text)] = vector

        return [vec if vec is not None else [] for vec in results]
