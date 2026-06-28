"""Chat routes — SSE streaming RAG answers + conversation history."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import get_chat_service
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.search import ChatRequest
from app.services.chat_service import ChatService
from app.db.database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from utils.response import success_response

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Stream a grounded answer as Server-Sent Events."""

    async def event_source() -> AsyncIterator[str]:
        async for event in service.answer_stream(
            payload.message, payload.conversation_id, payload.folder_id
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.get("/conversations")
async def list_conversations(
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    repo = ConversationRepository(session)
    conversations = await repo.list_all()
    data = [
        {"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()}
        for c in conversations
    ]
    return success_response(message="Conversations retrieved", data=data)


@router.get("/conversations/{conversation_id}/messages")
async def list_messages(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    repo = ConversationRepository(session)
    messages = await repo.list_messages(conversation_id)
    data = [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "cited_chunk_ids": m.cited_chunk_ids,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]
    return success_response(message="Messages retrieved", data=data)
