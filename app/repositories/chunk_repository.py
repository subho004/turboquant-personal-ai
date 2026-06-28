"""Chunk data access, including the FTS5 keyword mirror.

Keeps the ``chunks`` table and the ``chunks_fts`` virtual table in sync
so retrieval can combine vector and keyword (BM25) results.
"""

from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.file import File
from app.services.chunker import ChunkDraft

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_many(self, file_id: int, drafts: list[ChunkDraft]) -> list[Chunk]:
        """Insert chunk rows + their FTS mirror; return rows with ids."""

        chunks = [
            Chunk(
                file_id=file_id,
                ordinal=d.ordinal,
                text=d.text,
                token_count=d.token_count,
                heading_path=d.heading_path,
            )
            for d in drafts
        ]
        self._session.add_all(chunks)
        await self._session.flush()  # assigns ids
        for chunk in chunks:
            await self._session.execute(
                text("INSERT INTO chunks_fts(rowid, text) VALUES (:rid, :txt)"),
                {"rid": chunk.id, "txt": chunk.text},
            )
        return chunks

    async def get_by_ids(self, ids: list[int]) -> list[Chunk]:
        if not ids:
            return []
        result = await self._session.execute(select(Chunk).where(Chunk.id.in_(ids)))
        return list(result.scalars().all())

    async def details_by_ids(self, ids: list[int]) -> dict[int, tuple[Chunk, str]]:
        """Map ``chunk_id -> (Chunk, file_name)`` for the given ids."""

        if not ids:
            return {}
        result = await self._session.execute(
            select(Chunk, File.name)
            .join(File, File.id == Chunk.file_id)
            .where(Chunk.id.in_(ids))
        )
        return {chunk.id: (chunk, name) for chunk, name in result.all()}

    async def ids_by_file(self, file_id: int) -> list[int]:
        result = await self._session.execute(
            select(Chunk.id).where(Chunk.file_id == file_id)
        )
        return [row[0] for row in result.all()]

    async def ids_by_folders(self, folder_ids: list[int]) -> list[int]:
        if not folder_ids:
            return []
        result = await self._session.execute(
            select(Chunk.id)
            .join(File, File.id == Chunk.file_id)
            .where(File.folder_id.in_(folder_ids))
        )
        return [row[0] for row in result.all()]

    async def ids_by_date_range(self, start: datetime, end: datetime) -> list[int]:
        result = await self._session.execute(
            select(Chunk.id)
            .join(File, File.id == Chunk.file_id)
            .where(File.created_at >= start, File.created_at <= end)
        )
        return [row[0] for row in result.all()]

    async def delete_by_file(self, file_id: int) -> list[int]:
        ids = await self.ids_by_file(file_id)
        for cid in ids:
            await self._session.execute(
                text("DELETE FROM chunks_fts WHERE rowid = :rid"), {"rid": cid}
            )
        if ids:
            await self._session.execute(delete(Chunk).where(Chunk.id.in_(ids)))
        return ids

    async def keyword_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """BM25 keyword search; returns ``[(chunk_id, rank_score)]``."""

        terms = _WORD_RE.findall(query)
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms)
        result = await self._session.execute(
            text(
                "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH :q ORDER BY score LIMIT :k"
            ),
            {"q": match, "k": limit},
        )
        # bm25 returns lower = better; flip sign so higher = better downstream.
        return [(int(row[0]), -float(row[1])) for row in result.all()]

    async def count(self) -> int:
        result = await self._session.execute(select(Chunk.id))
        return len(result.all())
