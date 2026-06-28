"""Health check routes.

Exposes a liveness endpoint at `/health`. The base path `/` serves the
web UI (mounted in the app factory), so probes use `/health`.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils.response import success_response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> JSONResponse:
    """Return service liveness status."""

    return success_response(data={"status": "ok"}, message="healthy")
