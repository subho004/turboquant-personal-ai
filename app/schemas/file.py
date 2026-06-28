"""File request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    folder_id: int
    name: str
    ext: str
    size_bytes: int
    status: str
    error: str | None
    num_chunks: int
    created_at: datetime
    indexed_at: datetime | None


class FilePreviewResponse(BaseModel):
    id: int
    name: str
    text: str
