"""Memory ORM model.

``Memory.id`` is the external id stored in the TurboVec *memory* index.
Ids are offset into a disjoint range (``settings.memory_id_offset``) so
they never collide with chunk ids.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base

KIND_CHAT_SUMMARY = "chat_summary"
KIND_FACT = "fact"
KIND_SEARCH = "search"


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default=KIND_FACT)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
