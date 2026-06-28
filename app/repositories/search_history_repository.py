"""Search-history data access — every search becomes memory."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.search_history import SearchHistory


class SearchHistoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, query: str, top_chunk_ids: list[int], answer_preview: str
    ) -> SearchHistory:
        row = SearchHistory(
            query=query,
            top_chunk_ids=top_chunk_ids,
            answer_preview=answer_preview[:1024],
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_recent(self, limit: int = 20) -> list[SearchHistory]:
        result = await self._session.execute(
            select(SearchHistory).order_by(SearchHistory.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
