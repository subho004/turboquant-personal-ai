"""Process-wide token & cost meter.

Embedding and chat providers report token usage here; the UI polls
``/api/v1/usage`` to show a running spend total. Prices are configurable
via settings (defaults in USD per million tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings


@dataclass
class Usage:
    embed_tokens: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    requests: int = 0

    @property
    def cost_usd(self) -> float:
        return round(
            self.embed_tokens / 1_000_000 * settings.price_embed_per_mtok
            + self.chat_input_tokens / 1_000_000 * settings.price_chat_in_per_mtok
            + self.chat_output_tokens / 1_000_000 * settings.price_chat_out_per_mtok,
            6,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "embed_tokens": self.embed_tokens,
            "chat_input_tokens": self.chat_input_tokens,
            "chat_output_tokens": self.chat_output_tokens,
            "total_tokens": self.embed_tokens
            + self.chat_input_tokens
            + self.chat_output_tokens,
            "requests": self.requests,
            "cost_usd": self.cost_usd,
        }


class UsageMeter:
    def __init__(self) -> None:
        self._usage = Usage()

    def add_embeddings(self, tokens: int) -> None:
        self._usage.embed_tokens += tokens
        self._usage.requests += 1

    def add_chat(self, input_tokens: int, output_tokens: int) -> None:
        self._usage.chat_input_tokens += input_tokens
        self._usage.chat_output_tokens += output_tokens
        self._usage.requests += 1

    def snapshot(self) -> dict[str, float | int]:
        return self._usage.as_dict()


@lru_cache(maxsize=1)
def get_usage_meter() -> UsageMeter:
    return UsageMeter()
