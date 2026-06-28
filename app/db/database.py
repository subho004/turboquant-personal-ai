"""Async database engine, session factory, and schema initialisation.

Uses SQLAlchemy 2.x async with the `aiosqlite` driver. Sessions are
provided to routes via the `get_session` dependency — never created
inside services (see the engineering guide).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


engine = create_async_engine(settings.database_url, echo=False, future=True)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session that commits on success.

    Commits when the request handler returns cleanly and rolls back on
    any error, so routes and services never manage the transaction
    boundary themselves.
    """

    async with SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create tables and the FTS5 virtual table used for keyword search.

    Imported here (not at module top) so all models register on `Base`
    before `create_all` runs.
    """

    # Import every model module so it registers on Base.metadata.
    from app.models import (  # noqa: F401
        chunk,
        conversation,
        file,
        folder,
        memory,
        message,
        search_history,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # FTS5 mirror of chunk text for hybrid (keyword + vector) retrieval.
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(text, content='chunks', content_rowid='id')"
            )
        )
    logger.info("Database initialised at %s", settings.database_url)
