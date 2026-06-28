"""FastAPI dependency providers.

Composes repositories and services from a request-scoped DB session and
process-wide singletons (embedder, LLM, parser, vector store). Routes
depend on these via ``Depends`` — never instantiate services inline.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.file_repository import FileRepository
from app.repositories.folder_repository import FolderRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.search_history_repository import SearchHistoryRepository
from app.services.benchmark_service import BenchmarkService
from app.services.chat_service import ChatService
from app.services.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from app.services.file_service import FileService
from app.services.folder_service import FolderService
from app.services.ingest import IngestService
from app.services.llm import LLMProvider, OpenAILLMProvider
from app.services.memory_service import MemoryService
from app.services.parser import Parser
from app.services.reranker import Reranker
from app.services.retrieval import RetrievalService
from app.services.search_service import SearchService
from app.services.vector_store import VectorStore, get_vector_store


# --- process-wide singletons -------------------------------------------------
@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    return OpenAIEmbeddingProvider()  # holds an in-memory content-hash cache


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    return OpenAILLMProvider()


@lru_cache(maxsize=1)
def get_parser() -> Parser:
    return Parser()


# --- request-scoped services -------------------------------------------------
def get_folder_service(
    session: AsyncSession = Depends(get_session),
) -> FolderService:
    return FolderService(FolderRepository(session))


def get_file_service(
    session: AsyncSession = Depends(get_session),
) -> FileService:
    files = FileRepository(session)
    folders = FolderRepository(session)
    chunks = ChunkRepository(session)
    store: VectorStore = get_vector_store()
    ingest = IngestService(files, chunks, get_parser(), get_embedder(), store)
    return FileService(files, folders, chunks, ingest, store)


def _retrieval(session: AsyncSession) -> RetrievalService:
    return RetrievalService(
        get_embedder(),
        get_vector_store(),
        ChunkRepository(session),
        FolderRepository(session),
        MemoryRepository(session),
        reranker=Reranker(get_llm()),
    )


def get_search_service(
    session: AsyncSession = Depends(get_session),
) -> SearchService:
    return SearchService(
        _retrieval(session), SearchHistoryRepository(session), get_llm()
    )


def get_chat_service(
    session: AsyncSession = Depends(get_session),
) -> ChatService:
    memory = MemoryService(
        MemoryRepository(session), get_embedder(), get_vector_store(), get_llm()
    )
    return ChatService(
        session,
        _retrieval(session),
        get_llm(),
        ConversationRepository(session),
        SearchHistoryRepository(session),
        memory,
    )


def get_benchmark_service() -> BenchmarkService:
    return BenchmarkService()
