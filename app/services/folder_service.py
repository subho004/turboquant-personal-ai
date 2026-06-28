"""Folder use-cases."""

from __future__ import annotations

from app.core.exceptions import NotFoundError
from app.models.folder import Folder
from app.repositories.folder_repository import FolderRepository


class FolderService:
    def __init__(self, repo: FolderRepository) -> None:
        self._repo = repo

    async def create(self, name: str, parent_id: int | None) -> Folder:
        if parent_id is not None and await self._repo.get(parent_id) is None:
            raise NotFoundError(f"Parent folder '{parent_id}' not found")
        return await self._repo.create(name=name, parent_id=parent_id)

    async def list_all(self) -> list[Folder]:
        return await self._repo.list_all()

    async def get_or_404(self, folder_id: int) -> Folder:
        folder = await self._repo.get(folder_id)
        if folder is None:
            raise NotFoundError(f"Folder '{folder_id}' not found")
        return folder
