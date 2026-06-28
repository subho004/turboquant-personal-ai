"""TurboVec vector store.

Wraps two `IdMapIndex` instances — one for document chunks, one for
conversation memories — behind a single async writer lock. Tracks which
ids are present so allowlist searches never raise ``KeyError``, and
persists each index atomically (temp file + rename).
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from turbovec import IdMapIndex

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

MAIN = "main"
MEMORY = "memory"
_BIT_WIDTH = 4


class _Index:
    """A single TurboVec index plus its on-disk path and id set."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.ids: set[int] = set()
        if path.exists():
            self.index = IdMapIndex.load(str(path))
            logger.info("Loaded TurboVec index from %s", path)
        else:
            self.index = IdMapIndex(dim=settings.embed_dim, bit_width=_BIT_WIDTH)

    def persist(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.index.write(str(tmp))
        os.replace(tmp, self.path)


class VectorStore:
    """Thread-/task-safe facade over the document and memory indexes."""

    def __init__(self) -> None:
        index_dir = Path(settings.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._indexes: dict[str, _Index] = {
            MAIN: _Index(index_dir / "main.tvim"),
            MEMORY: _Index(index_dir / "memory.tvim"),
        }

    @staticmethod
    def _as_matrix(vectors: list[list[float]]) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))

    async def add(
        self, vectors: list[list[float]], ids: list[int], index: str = MAIN
    ) -> None:
        """Add vectors with stable external ids (idempotent on re-add)."""

        if not vectors:
            return
        target = self._indexes[index]
        async with self._lock:
            # Drop ids already present so re-indexing never raises.
            fresh = [(v, i) for v, i in zip(vectors, ids) if i not in target.ids]
            if not fresh:
                return
            mat = self._as_matrix([v for v, _ in fresh])
            id_arr = np.asarray([i for _, i in fresh], dtype=np.uint64)
            target.index.add_with_ids(mat, id_arr)
            target.ids.update(int(i) for i in id_arr)
            target.persist()

    async def remove(self, ids: list[int], index: str = MAIN) -> None:
        """Remove ids that are present; missing ids are ignored."""

        target = self._indexes[index]
        async with self._lock:
            removed = False
            for i in ids:
                if i in target.ids:
                    target.index.remove(int(i))
                    target.ids.discard(i)
                    removed = True
            if removed:
                target.persist()

    async def search(
        self,
        query: list[float],
        k: int,
        allowed_ids: list[int] | None = None,
        index: str = MAIN,
    ) -> list[tuple[int, float]]:
        """Return ``[(id, score)]`` best matches, optionally id-filtered."""

        target = self._indexes[index]
        async with self._lock:
            if not target.ids:
                return []
            allowlist: np.ndarray | None = None
            if allowed_ids is not None:
                present = [i for i in allowed_ids if i in target.ids]
                if not present:
                    return []
                allowlist = np.asarray(present, dtype=np.uint64)
            q = self._as_matrix([query])
            scores, result_ids = target.index.search(q, k=k, allowlist=allowlist)
        return [
            (int(rid), float(score)) for rid, score in zip(result_ids[0], scores[0])
        ]

    def count(self, index: str = MAIN) -> int:
        return len(self._indexes[index].ids)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the process-wide vector store singleton."""

    return VectorStore()
