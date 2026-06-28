"""File routes — upload, list, preview, reindex, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import JSONResponse

from app.api.deps import get_file_service
from app.schemas.file import FilePreviewResponse, FileResponse
from app.services.file_service import FileService
from utils.response import success_response

router = APIRouter(prefix="/api/v1", tags=["files"])


@router.post("/folders/{folder_id}/files", status_code=201)
async def upload_file(
    folder_id: int,
    file: UploadFile,
    service: FileService = Depends(get_file_service),
) -> JSONResponse:
    content = await file.read()
    record = await service.upload(folder_id, file.filename or "upload", content)
    data = FileResponse.model_validate(record)
    return success_response(message="File uploaded", data=data, status_code=201)


@router.get("/folders/{folder_id}/files")
async def list_files(
    folder_id: int,
    service: FileService = Depends(get_file_service),
) -> JSONResponse:
    files = await service.list_by_folder(folder_id)
    data = [FileResponse.model_validate(f) for f in files]
    return success_response(message="Files retrieved", data=data)


@router.get("/files/{file_id}/preview")
async def preview_file(
    file_id: int,
    service: FileService = Depends(get_file_service),
) -> JSONResponse:
    record = await service.get_or_404(file_id)
    text = await service.preview_text(file_id)
    data = FilePreviewResponse(id=record.id, name=record.name, text=text)
    return success_response(message="Preview retrieved", data=data)


@router.post("/files/{file_id}/reindex")
async def reindex_file(
    file_id: int,
    service: FileService = Depends(get_file_service),
) -> JSONResponse:
    record = await service.reindex(file_id)
    data = FileResponse.model_validate(record)
    return success_response(message="File reindexed", data=data)


@router.delete("/files/{file_id}/delete")
async def delete_file(
    file_id: int,
    service: FileService = Depends(get_file_service),
) -> JSONResponse:
    await service.delete(file_id)
    return success_response(message="File deleted")
