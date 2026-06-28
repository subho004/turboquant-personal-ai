"""Search use-cases: semantic search, scoped summaries, and second-brain
queries built on the same hybrid-retrieval primitive.
"""

from __future__ import annotations

from app.repositories.search_history_repository import SearchHistoryRepository
from app.schemas.search import SearchResponse, SourceItem
from app.services.llm import LLMProvider
from app.services.retrieval import RetrievedChunk, RetrievalService

_SNIPPET_CHARS = 240
_SUMMARISE_SYSTEM = (
    "Summarise the user's knowledge on the topic using only the numbered "
    "sources. Cite inline like [S1]. Be concise and well-structured."
)


def _to_source(chunk: RetrievedChunk, label: str) -> SourceItem:
    return SourceItem(
        chunk_id=chunk.chunk_id,
        file_id=chunk.file_id,
        file_name=chunk.file_name,
        heading_path=chunk.heading_path,
        score=round(chunk.score, 4),
        snippet=chunk.text[:_SNIPPET_CHARS],
    )


class SearchService:
    def __init__(
        self,
        retrieval: RetrievalService,
        history: SearchHistoryRepository,
        llm: LLMProvider,
    ) -> None:
        self._retrieval = retrieval
        self._history = history
        self._llm = llm

    async def search(
        self, query: str, folder_id: int | None, top_k: int
    ) -> SearchResponse:
        result = await self._retrieval.retrieve(query, folder_id=folder_id, top_k=top_k)
        sources = [_to_source(c, f"S{i + 1}") for i, c in enumerate(result.chunks)]
        await self._history.create(
            query, [s.chunk_id for s in sources[:10]], answer_preview=""
        )
        return SearchResponse(query=query, sources=sources)

    async def which_files_mention(self, query: str) -> list[dict[str, object]]:
        """Group hits by file — answers 'which files mention X?'."""

        result = await self._retrieval.retrieve(query, top_k=30)
        counts: dict[int, int] = {}
        meta: dict[int, tuple[str, str]] = {}
        for chunk in result.chunks:
            counts[chunk.file_id] = counts.get(chunk.file_id, 0) + 1
            meta.setdefault(
                chunk.file_id, (chunk.file_name, chunk.text[:_SNIPPET_CHARS])
            )
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {
                "file_id": file_id,
                "file_name": meta[file_id][0],
                "hits": hits,
                "snippet": meta[file_id][1],
            }
            for file_id, hits in ranked
        ]

    async def summarise(self, query: str, folder_id: int | None) -> dict[str, object]:
        """Retrieve + LLM-summarise — powers project mode & daily-note recall."""

        result = await self._retrieval.retrieve(query, folder_id=folder_id, top_k=12)
        if not result.chunks:
            return {"summary": "No relevant content found.", "sources": []}
        blocks = [
            f"[S{i + 1}] {c.file_name}: {c.text}" for i, c in enumerate(result.chunks)
        ]
        summary = await self._llm.complete(
            [
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {
                    "role": "user",
                    "content": "\n\n".join(blocks) + f"\n\nTopic: {query}",
                },
            ]
        )
        sources = [_to_source(c, f"S{i + 1}") for i, c in enumerate(result.chunks)]
        return {"summary": summary, "sources": [s.model_dump() for s in sources]}

    async def recent_searches(self, limit: int = 20) -> list[dict[str, object]]:
        rows = await self._history.list_recent(limit)
        return [
            {"query": r.query, "created_at": r.created_at.isoformat()} for r in rows
        ]
