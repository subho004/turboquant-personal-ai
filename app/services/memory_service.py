"""Conversation memory.

Stores short summaries / facts in the dedicated TurboVec memory index so
the assistant can recall them across conversations. Memory ids reuse the
``Memory`` row id (the memory index is separate from the chunk index, so
no collision is possible).
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.models.memory import KIND_CHAT_SUMMARY, KIND_FACT
from app.repositories.memory_repository import MemoryRepository
from app.services.embeddings import EmbeddingProvider
from app.services.llm import LLMProvider
from app.services.vector_store import MEMORY, VectorStore

logger = get_logger(__name__)

_SUMMARY_SYSTEM = (
    "Summarise the exchange below as a single factual sentence written so it "
    "can be recalled later out of context. Reply with only that sentence."
)


class MemoryService:
    def __init__(
        self,
        memories: MemoryRepository,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
    ) -> None:
        self._memories = memories
        self._embedder = embedder
        self._store = store
        self._llm = llm

    async def remember(
        self, text: str, kind: str = KIND_FACT, conversation_id: int | None = None
    ) -> None:
        memory = await self._memories.create(text, kind, conversation_id)
        vector = (await self._embedder.embed([text]))[0]
        await self._store.add([vector], [memory.id], index=MEMORY)

    async def summarise_exchange(
        self, conversation_id: int, question: str, answer: str
    ) -> None:
        """Best-effort: store a one-line summary of a Q/A exchange."""

        try:
            summary = await self._llm.complete(
                [
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"User: {question}\nAssistant: {answer}",
                    },
                ]
            )
            if summary:
                await self.remember(summary, KIND_CHAT_SUMMARY, conversation_id)
        except Exception:  # noqa: BLE001 — memory is best-effort, never fail the chat
            logger.exception("Failed to summarise exchange for memory")
