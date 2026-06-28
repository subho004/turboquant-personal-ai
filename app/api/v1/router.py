"""Aggregate API router.

Collects all v1 feature routers into a single router that the app
factory mounts in `main.py`.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.benchmark import router as benchmark_router
from app.api.v1.chat import router as chat_router
from app.api.v1.files import router as files_router
from app.api.v1.folders import router as folders_router
from app.api.v1.health import router as health_router
from app.api.v1.search import router as search_router
from app.api.v1.usage import router as usage_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(folders_router)
api_router.include_router(files_router)
api_router.include_router(chat_router)
api_router.include_router(search_router)
api_router.include_router(benchmark_router)
api_router.include_router(usage_router)
