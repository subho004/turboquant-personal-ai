"""Folder data access."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.folder import Folder


class FolderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, name: str, parent_id: int | None) -> Folder:
        folder = Folder(name=name, parent_id=parent_id)
        self._session.add(folder)
        await self._session.flush()
        return folder

    async def get(self, folder_id: int) -> Folder | None:
        return await self._session.get(Folder, folder_id)

    async def list_all(self) -> list[Folder]:
        result = await self._session.execute(select(Folder).order_by(Folder.name))
        return list(result.scalars().all())

    async def subtree_ids(self, folder_id: int) -> list[int]:
        """Return ``folder_id`` plus all descendant folder ids."""

        folders = await self.list_all()
        children: dict[int, list[int]] = {}
        for folder in folders:
            if folder.parent_id is not None:
                children.setdefault(folder.parent_id, []).append(folder.id)

        collected: list[int] = []
        stack = [folder_id]
        while stack:
            current = stack.pop()
            collected.append(current)
            stack.extend(children.get(current, []))
        return collected
