"""Search + chat request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    folder_id: int | None = None
    top_k: int = Field(default=12, ge=1, le=50)


class SourceItem(BaseModel):
    chunk_id: int
    file_id: int
    file_name: str
    heading_path: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    sources: list[SourceItem]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: int | None = None
    folder_id: int | None = None
