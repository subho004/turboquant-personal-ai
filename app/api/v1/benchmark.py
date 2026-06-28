"""Benchmark route — TurboVec vs FAISS on a synthetic corpus."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.api.deps import get_benchmark_service
from app.services.benchmark_service import BenchmarkService
from utils.response import success_response

router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])


@router.post("/run")
async def run_benchmark(
    n_vectors: int = Query(default=20_000, ge=1_000, le=50_000),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> JSONResponse:
    data = await service.run(n_vectors=n_vectors)
    return success_response(message="Benchmark complete", data=data)
