"""Usage route — running token + cost totals for the UI meter."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.usage import get_usage_meter
from utils.response import success_response

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


@router.get("/totals")
async def usage_totals() -> JSONResponse:
    return success_response(message="Usage retrieved", data=get_usage_meter().snapshot())
