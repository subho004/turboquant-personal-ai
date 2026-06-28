"""Search routes — semantic search + second-brain queries."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.api.deps import get_search_service
from app.schemas.search import SearchRequest
from app.services.search_service import SearchService
from utils.response import success_response

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("/query")
async def search_query(
    payload: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> JSONResponse:
    result = await service.search(payload.query, payload.folder_id, payload.top_k)
    return success_response(message="Search complete", data=result)


@router.get("/mentions")
async def which_files_mention(
    q: str = Query(min_length=1),
    service: SearchService = Depends(get_search_service),
) -> JSONResponse:
    data = await service.which_files_mention(q)
    return success_response(message="Mentions retrieved", data=data)


@router.post("/summarise")
async def summarise(
    payload: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> JSONResponse:
    data = await service.summarise(payload.query, payload.folder_id)
    return success_response(message="Summary generated", data=data)


@router.get("/history")
async def recent_searches(
    service: SearchService = Depends(get_search_service),
) -> JSONResponse:
    data = await service.recent_searches()
    return success_response(message="History retrieved", data=data)
