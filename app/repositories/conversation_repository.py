"""Conversation + message data access (one chat aggregate)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import Message


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, title: str = "New chat") -> Conversation:
        conversation = Conversation(title=title)
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get(self, conversation_id: int) -> Conversation | None:
        return await self._session.get(Conversation, conversation_id)

    async def list_all(self) -> list[Conversation]:
        result = await self._session.execute(
            select(Conversation).order_by(Conversation.created_at.desc())
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        cited_chunk_ids: list[int] | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            cited_chunk_ids=cited_chunk_ids or [],
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def list_messages(self, conversation_id: int) -> list[Message]:
        result = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at, Message.id)
        )
        return list(result.scalars().all())
