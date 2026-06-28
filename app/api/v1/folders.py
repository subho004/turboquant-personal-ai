"""Folder routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import get_folder_service
from app.schemas.folder import FolderCreate, FolderResponse
from app.services.folder_service import FolderService
from utils.response import success_response

router = APIRouter(prefix="/api/v1/folders", tags=["folders"])


@router.post("", status_code=201)
async def create_folder(
    payload: FolderCreate,
    service: FolderService = Depends(get_folder_service),
) -> JSONResponse:
    folder = await service.create(payload.name, payload.parent_id)
    data = FolderResponse.model_validate(folder)
    return success_response(message="Folder created", data=data, status_code=201)


@router.get("")
async def list_folders(
    service: FolderService = Depends(get_folder_service),
) -> JSONResponse:
    folders = await service.list_all()
    data = [FolderResponse.model_validate(f) for f in folders]
    return success_response(message="Folders retrieved", data=data)
