"""Hybrid retrieval — vector + keyword search fused with RRF.

Combines TurboVec semantic hits with SQLite FTS5 BM25 hits using
Reciprocal Rank Fusion, resolves optional folder scoping into an
allowlist, and also pulls relevant conversation memories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.config import settings
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.folder_repository import FolderRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import MAIN, MEMORY, VectorStore

if TYPE_CHECKING:
    from app.services.reranker import Reranker

_RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: int
    file_id: int
    file_name: str
    heading_path: str
    text: str
    score: float


@dataclass
class RetrievedMemory:
    memory_id: int
    text: str
    score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    memories: list[RetrievedMemory]
    best_vector_score: float  # for answer-grounding decisions
    has_keyword_hit: bool  # an exact BM25 term match was found


def _rrf(rankings: list[list[int]]) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked id lists."""

    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return fused


class RetrievalService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        chunks: ChunkRepository,
        folders: FolderRepository,
        memories: MemoryRepository,
        reranker: "Reranker | None" = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunks = chunks
        self._folders = folders
        self._memories = memories
        self._reranker = reranker

    async def retrieve(
        self, query: str, folder_id: int | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        k = top_k or settings.retrieval_top_k
        query_vec = (await self._embedder.embed([query]))[0]

        allowed_ids: list[int] | None = None
        if folder_id is not None:
            folder_ids = await self._folders.subtree_ids(folder_id)
            allowed_ids = await self._chunks.ids_by_folders(folder_ids)

        vector_hits = await self._store.search(
            query_vec, k=settings.rerank_candidates, allowed_ids=allowed_ids, index=MAIN
        )
        keyword_hits = await self._chunks.keyword_search(
            query, limit=settings.rerank_candidates
        )

        best_vector_score = vector_hits[0][1] if vector_hits else 0.0
        fused = _rrf(
            [[cid for cid, _ in vector_hits], [cid for cid, _ in keyword_hits]]
        )
        # Keep a wide candidate set, then let the reranker pick the final top_k.
        candidate_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[
            : settings.rerank_candidates
        ]

        details = await self._chunks.details_by_ids(candidate_ids)
        candidates: list[RetrievedChunk] = []
        for cid in candidate_ids:
            if cid not in details:
                continue
            chunk, file_name = details[cid]
            candidates.append(
                RetrievedChunk(
                    chunk_id=cid,
                    file_id=chunk.file_id,
                    file_name=file_name,
                    heading_path=chunk.heading_path,
                    text=chunk.text,
                    score=fused[cid],
                )
            )

        if self._reranker is not None and settings.rerank_enabled:
            chunks = await self._reranker.rerank(query, candidates, k)
        else:
            chunks = candidates[:k]

        memories = await self._retrieve_memories(query_vec)
        return RetrievalResult(
            chunks=chunks,
            memories=memories,
            best_vector_score=best_vector_score,
            has_keyword_hit=bool(keyword_hits),
        )

    async def _retrieve_memories(self, query_vec: list[float]) -> list[RetrievedMemory]:
        hits = await self._store.search(
            query_vec, k=settings.memory_top_k, index=MEMORY
        )
        if not hits:
            return []
        rows = {
            m.id: m for m in await self._memories.get_by_ids([mid for mid, _ in hits])
        }
        return [
            RetrievedMemory(memory_id=mid, text=rows[mid].text, score=score)
            for mid, score in hits
            if mid in rows
        ]
