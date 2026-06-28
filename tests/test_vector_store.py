"""Unit tests for the TurboVec vector-store wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import settings
from app.services.vector_store import MAIN, MEMORY, VectorStore


def _vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(settings.embed_dim).tolist()


@pytest.mark.asyncio
async def test_add_search_remove_roundtrip() -> None:
    store = VectorStore()
    vectors = [_vec(1), _vec(2), _vec(3)]
    await store.add(vectors, [10, 20, 30])

    assert store.count() == 3
    hits = await store.search(vectors[0], k=1)
    assert hits[0][0] == 10  # nearest to itself

    await store.remove([20])
    assert store.count() == 2


@pytest.mark.asyncio
async def test_allowlist_restricts_results() -> None:
    store = VectorStore()
    await store.add([_vec(1), _vec(2), _vec(3)], [10, 20, 30])

    hits = await store.search(_vec(1), k=5, allowed_ids=[20, 30])
    returned = {cid for cid, _ in hits}
    assert returned.issubset({20, 30})


@pytest.mark.asyncio
async def test_unknown_allowlist_id_is_ignored() -> None:
    store = VectorStore()
    await store.add([_vec(1)], [10])

    # id 999 is not in the index — must not raise KeyError.
    hits = await store.search(_vec(1), k=3, allowed_ids=[10, 999])
    assert hits[0][0] == 10


@pytest.mark.asyncio
async def test_re_add_is_idempotent() -> None:
    store = VectorStore()
    await store.add([_vec(1)], [10])
    await store.add([_vec(1)], [10])  # same id again must not raise
    assert store.count() == 1


@pytest.mark.asyncio
async def test_memory_index_is_separate() -> None:
    store = VectorStore()
    await store.add([_vec(1)], [10], index=MAIN)
    await store.add([_vec(2)], [10], index=MEMORY)
    assert store.count(MAIN) == 1
    assert store.count(MEMORY) == 1
