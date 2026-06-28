"""Ingestion pipeline: parse -> chunk -> embed -> index.

Drives one file through the per-file status state machine and keeps
SQLite, the FTS mirror, and the TurboVec index consistent. Runs
synchronously within the upload request so the response reflects the
final indexed state (simple and crash-evident for the MVP).
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.logging import get_logger
from app.models.file import (
    STATUS_EMBEDDING,
    STATUS_ERROR,
    STATUS_PARSING,
    File,
)
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.file_repository import FileRepository
from app.services.chunker import chunk_markdown
from app.services.embeddings import EmbeddingProvider
from app.services.parser import Parser
from app.services.vector_store import VectorStore

logger = get_logger(__name__)


class IngestService:
    def __init__(
        self,
        files: FileRepository,
        chunks: ChunkRepository,
        parser: Parser,
        embedder: EmbeddingProvider,
        store: VectorStore,
    ) -> None:
        self._files = files
        self._chunks = chunks
        self._parser = parser
        self._embedder = embedder
        self._store = store

    async def ingest(self, file: File, abs_path: Path) -> File:
        """Parse, chunk, embed and index a single file."""

        start = time.perf_counter()
        logger.info("Ingest started for file %s (%s)", file.id, file.name)
        try:
            await self._files.set_status(file, STATUS_PARSING)
            markdown = await self._parser.parse(abs_path, file.ext)

            drafts = chunk_markdown(markdown)
            if not drafts:
                await self._files.mark_indexed(file, len(markdown), 0)
                return file

            await self._files.set_status(file, STATUS_EMBEDDING)
            vectors = await self._embedder.embed([d.text for d in drafts])

            chunk_rows = await self._chunks.create_many(file.id, drafts)
            await self._store.add(vectors, [c.id for c in chunk_rows])

            await self._files.mark_indexed(file, len(markdown), len(chunk_rows))
            logger.info(
                "Ingest done for file %s: %d chunks in %.3fs",
                file.id,
                len(chunk_rows),
                time.perf_counter() - start,
            )
            return file
        except Exception as exc:  # noqa: BLE001 — record reason, surface in UI
            logger.exception("Ingest failed for file %s", file.id)
            await self._files.set_status(file, STATUS_ERROR, error=str(exc)[:1024])
            return file

    async def reindex(self, file: File, abs_path: Path) -> File:
        """Remove a file's existing chunks, then ingest fresh."""

        old_ids = await self._chunks.delete_by_file(file.id)
        if old_ids:
            await self._store.remove(old_ids)
        return await self.ingest(file, abs_path)
