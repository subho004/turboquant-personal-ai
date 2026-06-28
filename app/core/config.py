"""Application configuration.

Exposes a typed `Settings` object loaded from environment variables
(and an optional `.env` file) via `pydantic-settings`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Values are read from the environment (case-insensitive) and from a
    `.env` file if present. Unknown keys in `.env` are ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Personal AI (TurboVec)"
    env: str = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = ["*"]
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # Filesystem root for uploads + on-disk TurboVec indexes
    data_dir: str = "./data"

    # OpenAI
    openai_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    # gpt-5.5-mini is not a released model id; gpt-5-mini is the closest small
    # GPT-5 chat model available. Override via CHAT_MODEL when it ships.
    chat_model: str = "gpt-5-mini"
    transcribe_model: str = "whisper-1"

    # Retrieval / chunking
    chunk_tokens: int = 500
    chunk_overlap: int = 80
    retrieval_top_k: int = 12
    memory_top_k: int = 6
    rerank_candidates: int = 30
    rerank_enabled: bool = True  # LLM rerank of candidates down to top_k
    min_score: float = 0.15  # below this, the assistant answers "I don't know"

    # Token pricing (USD per 1M tokens) for the usage/cost meter — override per
    # model via env. Defaults approximate text-embedding-3-small + gpt-5-mini.
    price_embed_per_mtok: float = 0.02
    price_chat_in_per_mtok: float = 0.25
    price_chat_out_per_mtok: float = 2.00

    # Memory ids live in a disjoint range so a uint64 always maps to one source
    memory_id_offset: int = 1 << 40

    @property
    def uploads_dir(self) -> str:
        return f"{self.data_dir}/uploads"

    @property
    def index_dir(self) -> str:
        return f"{self.data_dir}/index"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Allow a comma-separated string for `CORS_ORIGINS` env vars."""

        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


# Single settings instance used across the app
settings = Settings()


__all__ = ["Settings", "settings"]
