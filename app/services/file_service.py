"""File use-cases: upload, list, preview, reindex, delete.

Validates and sanitises uploads (extension allowlist, size cap,
path-traversal-safe names) before handing off to the ingestion pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import (
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)
from app.models.file import STATUS_PENDING, File
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.file_repository import FileRepository
from app.repositories.folder_repository import FolderRepository
from app.services.ingest import IngestService
from app.services.parser import SUPPORTED_EXTS
from app.services.vector_store import VectorStore
from utils.hashing import sha256_bytes

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]")


def _safe_name(filename: str) -> str:
    """Strip directories and unsafe characters from an upload filename."""

    base = Path(filename).name  # drops any path components / traversal
    cleaned = _UNSAFE.sub("_", base).strip()
    return cleaned or "upload"


class FileService:
    def __init__(
        self,
        files: FileRepository,
        folders: FolderRepository,
        chunks: ChunkRepository,
        ingest: IngestService,
        store: VectorStore,
    ) -> None:
        self._files = files
        self._folders = folders
        self._chunks = chunks
        self._ingest = ingest
        self._store = store

    async def upload(self, folder_id: int, filename: str, content: bytes) -> File:
        if await self._folders.get(folder_id) is None:
            raise NotFoundError(f"Folder '{folder_id}' not found")
        if not content:
            raise BadRequestError("Empty file")
        if len(content) > MAX_UPLOAD_BYTES:
            raise BadRequestError("File exceeds the 25 MB limit")

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise UnprocessableEntityError(f"Unsupported file type '{ext}'")

        content_hash = sha256_bytes(content)
        existing = await self._files.find_by_hash(folder_id, content_hash)
        if existing is not None:
            return existing  # idempotent: identical upload already indexed

        safe = _safe_name(filename)
        folder_dir = Path(settings.uploads_dir) / str(folder_id)
        folder_dir.mkdir(parents=True, exist_ok=True)
        abs_path = folder_dir / f"{content_hash[:8]}_{safe}"
        abs_path.write_bytes(content)

        file = File(
            folder_id=folder_id,
            name=safe,
            rel_path=str(abs_path.relative_to(settings.data_dir)),
            ext=ext,
            size_bytes=len(content),
            content_hash=content_hash,
            status=STATUS_PENDING,
        )
        await self._files.create(file)
        return await self._ingest.ingest(file, abs_path)

    async def list_by_folder(self, folder_id: int) -> list[File]:
        if await self._folders.get(folder_id) is None:
            raise NotFoundError(f"Folder '{folder_id}' not found")
        return await self._files.list_by_folder(folder_id)

    async def get_or_404(self, file_id: int) -> File:
        file = await self._files.get(file_id)
        if file is None:
            raise NotFoundError(f"File '{file_id}' not found")
        return file

    async def preview_text(self, file_id: int) -> str:
        file = await self.get_or_404(file_id)
        ids = await self._chunks.ids_by_file(file.id)
        chunks = await self._chunks.get_by_ids(ids)
        chunks.sort(key=lambda c: c.ordinal)
        return "\n\n".join(c.text for c in chunks)

    async def reindex(self, file_id: int) -> File:
        file = await self.get_or_404(file_id)
        abs_path = Path(settings.data_dir) / file.rel_path
        if not abs_path.exists():
            raise NotFoundError("Stored file is missing on disk")
        return await self._ingest.reindex(file, abs_path)

    async def delete(self, file_id: int) -> None:
        file = await self.get_or_404(file_id)
        old_ids = await self._chunks.delete_by_file(file.id)
        if old_ids:
            await self._store.remove(old_ids)
        abs_path = Path(settings.data_dir) / file.rel_path
        abs_path.unlink(missing_ok=True)
        await self._files.delete(file)
