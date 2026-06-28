"""Retrieval evaluation harness.

Ingests a small golden corpus through the REAL pipeline (MarkItDown +
OpenAI embeddings + TurboVec) and measures retrieval quality against a
hand-labelled question set:

  * hit@k   — did the expected file appear in the top-k sources?
  * MRR     — mean reciprocal rank of the expected file.

Run:  python scripts/evaluate.py    (needs OPENAI_API_KEY in .env)

Quality only depends on embeddings; no LLM calls are made, so it is cheap
and repeatable. All state lives in a temp dir and is removed on exit.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.db.database import Base  # noqa: E402
from app.models import (  # noqa: E402, F401  (register all tables on Base)
    chunk,
    conversation,
    file,
    folder,
    memory,
    message,
    search_history,
)
from app.repositories.chunk_repository import ChunkRepository  # noqa: E402
from app.repositories.file_repository import FileRepository  # noqa: E402
from app.repositories.folder_repository import FolderRepository  # noqa: E402
from app.repositories.memory_repository import MemoryRepository  # noqa: E402
from app.services.embeddings import OpenAIEmbeddingProvider  # noqa: E402
from app.services.file_service import FileService  # noqa: E402
from app.services.ingest import IngestService  # noqa: E402
from app.services.parser import Parser  # noqa: E402
from app.services.retrieval import RetrievalService  # noqa: E402
from app.services.vector_store import get_vector_store  # noqa: E402

logger = get_logger(__name__)

CORPUS: dict[str, str] = {
    "turbovec.md": (
        "# TurboVec\n\nTurboVec is a vector index built on TurboQuant. It compresses "
        "embeddings to 4 bits per dimension and searches faster than FAISS while using "
        "far less RAM. It supports online inserts without any training phase."
    ),
    "fraud.md": (
        "# Fraud Detection\n\nOur fraud model uses gradient boosted trees on transaction "
        "features. Precision at the top decile is the key business metric. We retrain weekly."
    ),
    "gnn.md": (
        "# Graph Neural Networks\n\nGNNs aggregate messages across edges of a graph. They "
        "are useful for node classification and link prediction on relational data."
    ),
    "journal.md": (
        "# March Journal\n\nThis month I focused on benchmarking vector databases and "
        "writing the quarterly fraud detection report. Also read three papers on attention."
    ),
}

GOLDEN: list[tuple[str, str]] = [
    ("How does TurboVec compress embeddings?", "turbovec.md"),
    ("What metric matters for the fraud model?", "fraud.md"),
    ("What do graph neural networks do with edges?", "gnn.md"),
    ("What was I working on in March?", "journal.md"),
    ("Which index searches faster than FAISS?", "turbovec.md"),
    ("How often is the fraud model retrained?", "fraud.md"),
]
TOP_K = 5


async def _build_corpus(file_service: FileService, folder_id: int) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="eval_src_"))
    try:
        for name, body in CORPUS.items():
            (tmp / name).write_text(body, encoding="utf-8")
            await file_service.upload(folder_id, name, (tmp / name).read_bytes())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def _evaluate() -> None:
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not set; cannot run evaluation")
        raise SystemExit(1)

    tmp_data = tempfile.mkdtemp(prefix="eval_data_")
    settings.data_dir = tmp_data
    get_vector_store.cache_clear()

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_data}/eval.db")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(text, content='chunks', content_rowid='id')"
            )
        )

    embedder = OpenAIEmbeddingProvider()
    store = get_vector_store()
    start = time.perf_counter()

    async with session_factory() as session:
        files = FileRepository(session)
        folders = FolderRepository(session)
        chunks = ChunkRepository(session)
        ingest = IngestService(files, chunks, Parser(), embedder, store)
        file_service = FileService(files, folders, chunks, ingest, store)

        folder = await folders.create("Eval", None)
        await _build_corpus(file_service, folder.id)
        await session.commit()
        logger.info("Ingested %d documents", len(CORPUS))

        retrieval = RetrievalService(
            embedder, store, chunks, folders, MemoryRepository(session)
        )

        hits = 0
        reciprocal_ranks = 0.0
        print("\n=== Retrieval evaluation (hybrid: TurboVec + BM25) ===\n")
        for question, expected in GOLDEN:
            result = await retrieval.retrieve(question, top_k=TOP_K)
            names = [c.file_name for c in result.chunks]
            rank = names.index(expected) + 1 if expected in names else 0
            if rank:
                hits += 1
                reciprocal_ranks += 1.0 / rank
            mark = "✓" if rank else "✗"
            print(f"  {mark}  rank={rank or '-'}  {question}")
            print(f"       expected={expected}  got={names[:3]}")

        n = len(GOLDEN)
        print(f"\n  hit@{TOP_K}: {hits}/{n} = {hits / n:.2%}")
        print(f"  MRR:    {reciprocal_ranks / n:.3f}")
        print(f"  elapsed: {time.perf_counter() - start:.2f}s\n")

    await engine.dispose()
    shutil.rmtree(tmp_data, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(_evaluate())
