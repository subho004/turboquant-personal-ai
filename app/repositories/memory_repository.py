"""Conversation-memory data access."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import Memory


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, text: str, kind: str, conversation_id: int | None) -> Memory:
        memory = Memory(text=text, kind=kind, conversation_id=conversation_id)
        self._session.add(memory)
        await self._session.flush()
        return memory

    async def get_by_ids(self, ids: list[int]) -> list[Memory]:
        if not ids:
            return []
        result = await self._session.execute(select(Memory).where(Memory.id.in_(ids)))
        return list(result.scalars().all())
