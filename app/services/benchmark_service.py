"""TurboVec vs FAISS benchmark.

Builds a synthetic corpus and compares TurboVec (4-bit) against FAISS
exact (IndexFlatIP) and FAISS compressed (IndexPQ) on build time, search
latency, on-disk size, and recall@k. CPU-bound work runs in a thread.
"""

from __future__ import annotations

import os
import tempfile
import time

import faiss
import numpy as np
from turbovec import IdMapIndex

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_MAX_VECTORS = 50_000  # keeps FP32 ground truth memory-safe at dim 1536
_PQ_SUBQUANTIZERS = 96  # must divide dim (1536 / 96 = 16 dims each)
_PQ_BITS = 8


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def _percentile_ms(latencies: list[float], pct: float) -> float:
    return round(float(np.percentile(latencies, pct)) * 1000, 3)


def _file_size_mb(write_fn, suffix: str) -> float:
    path = tempfile.mktemp(suffix=suffix)
    try:
        write_fn(path)
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    finally:
        if os.path.exists(path):
            os.remove(path)


def _search_latencies(search_fn, queries: np.ndarray) -> list[float]:
    latencies: list[float] = []
    for q in queries:
        start = time.perf_counter()
        search_fn(q)
        latencies.append(time.perf_counter() - start)
    return latencies


def _run(n_vectors: int, n_queries: int, k: int) -> dict[str, object]:
    dim = settings.embed_dim
    rng = np.random.default_rng(42)
    base = _normalise(rng.standard_normal((n_vectors, dim)))
    queries = _normalise(rng.standard_normal((n_queries, dim)))
    ids = np.arange(n_vectors, dtype=np.uint64)

    # --- FAISS exact (ground truth for recall) ---
    flat = faiss.IndexFlatIP(dim)
    flat.add(base)
    _, truth = flat.search(queries, k)

    # --- TurboVec 4-bit ---
    tv = IdMapIndex(dim=dim, bit_width=4)
    t0 = time.perf_counter()
    tv.add_with_ids(base, ids)
    tv_build = round(time.perf_counter() - t0, 3)
    tv_lat = _search_latencies(lambda q: tv.search(q[None, :], k=k), queries)
    tv_size = _file_size_mb(tv.write, ".tvim")
    tv_hits = 0
    for i, q in enumerate(queries):
        _, got = tv.search(q[None, :], k=k)
        tv_hits += len(set(got[0].tolist()) & set(truth[i].tolist()))

    # --- FAISS PQ (compressed competitor) ---
    pq = faiss.IndexPQ(dim, _PQ_SUBQUANTIZERS, _PQ_BITS)
    t0 = time.perf_counter()
    pq.train(base)
    pq.add(base)
    pq_build = round(time.perf_counter() - t0, 3)
    pq_lat = _search_latencies(lambda q: pq.search(q[None, :], k=k), queries)
    pq_size = _file_size_mb(lambda p: faiss.write_index(pq, p), ".faiss")
    pq_hits = 0
    for i, q in enumerate(queries):
        _, got = pq.search(q[None, :].copy(), k)
        pq_hits += len(set(got[0].tolist()) & set(truth[i].tolist()))

    fp32_mb = round(n_vectors * dim * 4 / (1024 * 1024), 2)
    denom = n_queries * k
    return {
        "config": {"n_vectors": n_vectors, "dim": dim, "n_queries": n_queries, "k": k},
        "fp32_ram_mb": fp32_mb,
        "results": [
            {
                "name": "TurboVec (4-bit)",
                "build_s": tv_build,
                "p50_ms": _percentile_ms(tv_lat, 50),
                "p95_ms": _percentile_ms(tv_lat, 95),
                "size_mb": tv_size,
                "compression_x": round(fp32_mb / tv_size, 1) if tv_size else None,
                "recall_at_k": round(tv_hits / denom, 4),
                "needs_training": False,
            },
            {
                "name": "FAISS IndexPQ",
                "build_s": pq_build,
                "p50_ms": _percentile_ms(pq_lat, 50),
                "p95_ms": _percentile_ms(pq_lat, 95),
                "size_mb": pq_size,
                "compression_x": round(fp32_mb / pq_size, 1) if pq_size else None,
                "recall_at_k": round(pq_hits / denom, 4),
                "needs_training": True,
            },
            {
                "name": "FAISS IndexFlat (exact)",
                "build_s": None,
                "p50_ms": _percentile_ms(
                    _search_latencies(lambda q: flat.search(q[None, :], k), queries), 50
                ),
                "p95_ms": None,
                "size_mb": fp32_mb,
                "compression_x": 1.0,
                "recall_at_k": 1.0,
                "needs_training": False,
            },
        ],
    }


class BenchmarkService:
    async def run(
        self, n_vectors: int = 20_000, n_queries: int = 200, k: int = 10
    ) -> dict[str, object]:
        import asyncio

        n_vectors = max(1_000, min(n_vectors, _MAX_VECTORS))
        logger.info("Benchmark started: n=%d", n_vectors)
        start = time.perf_counter()
        result = await asyncio.to_thread(_run, n_vectors, n_queries, k)
        result["elapsed_s"] = round(time.perf_counter() - start, 2)
        result["note"] = (
            "Synthetic random vectors. Size/latency/build are corpus-driven and "
            "representative; recall on random data is a lower bound. Hardware: "
            "TurboVec's edge is largest on Apple Silicon (NEON)."
        )
        logger.info("Benchmark done in %.2fs", result["elapsed_s"])
        return result
