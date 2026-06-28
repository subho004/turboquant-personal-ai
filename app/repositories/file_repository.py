"""File (uploaded document) data access."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file import STATUS_INDEXED, File


class FileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, file: File) -> File:
        self._session.add(file)
        await self._session.flush()
        return file

    async def get(self, file_id: int) -> File | None:
        return await self._session.get(File, file_id)

    async def find_by_hash(self, folder_id: int, content_hash: str) -> File | None:
        result = await self._session.execute(
            select(File).where(
                File.folder_id == folder_id, File.content_hash == content_hash
            )
        )
        return result.scalars().first()

    async def list_by_folder(self, folder_id: int) -> list[File]:
        result = await self._session.execute(
            select(File).where(File.folder_id == folder_id).order_by(File.created_at)
        )
        return list(result.scalars().all())

    async def list_all(self) -> list[File]:
        result = await self._session.execute(select(File).order_by(File.created_at))
        return list(result.scalars().all())

    async def set_status(
        self, file: File, status: str, error: str | None = None
    ) -> None:
        file.status = status
        file.error = error
        await self._session.flush()

    async def delete(self, file: File) -> None:
        await self._session.delete(file)
        await self._session.flush()

    async def mark_indexed(
        self, file: File, parsed_chars: int, num_chunks: int
    ) -> None:
        file.status = STATUS_INDEXED
        file.parsed_chars = parsed_chars
        file.num_chunks = num_chunks
        file.indexed_at = datetime.now(timezone.utc)
        file.error = None
        await self._session.flush()
