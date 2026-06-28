"""Shared pytest fixtures.

Wires the FastAPI app to an isolated temp database + temp TurboVec index
directory, and replaces the OpenAI-backed providers with deterministic
fakes so the suite runs offline, free, and repeatably. Every test gets a
fresh DB and index dir, and all temp state is cleaned up on teardown.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api import deps
from app.core.config import settings
from app.db.database import Base, get_session
from app.services.vector_store import get_vector_store
from main import app


class FakeEmbedder:
    """Deterministic embeddings: same text -> same vector, dim = embed_dim."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for value in texts:
            rng = np.random.default_rng(abs(hash(value)) % (2**32))
            vectors.append(rng.standard_normal(settings.embed_dim).tolist())
        return vectors


class FakeLLM:
    """Echoes a grounded-looking answer so chat flows are testable offline."""

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        for token in ["Based ", "on ", "[S1] ", "your ", "files."]:
            yield token

    async def complete(self, messages: list[dict[str, str]]) -> str:
        return "A concise test summary [S1]."


@pytest.fixture(autouse=True)
def _isolated_storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Temp data dir, fake providers, and a reset vector-store singleton."""

    tmp = tempfile.mkdtemp(prefix="turbovec_test_")
    monkeypatch.setattr(settings, "data_dir", tmp)
    monkeypatch.setattr(deps, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(deps, "get_llm", lambda: FakeLLM())
    get_vector_store.cache_clear()
    yield
    get_vector_store.cache_clear()
    shutil.rmtree(tmp, ignore_errors=True)


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    db_path = Path(settings.data_dir) / "test.db"
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(text, content='chunks', content_rowid='id')"
            )
        )

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = override_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()
