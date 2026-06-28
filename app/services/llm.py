"""LLM provider — GPT-powered answer generation and summarisation.

Defines a protocol so callers depend on an abstraction; the OpenAI
implementation streams chat answers and produces short summaries.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, cast, runtime_checkable

from openai.types.chat import ChatCompletionMessageParam

from app.core.config import settings
from app.services.openai_client import get_openai_client
from app.services.usage import get_usage_meter

ChatMessage = dict[str, str]


def _as_params(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    """Adapt plain role/content dicts to the SDK's typed message params."""

    return cast("list[ChatCompletionMessageParam]", messages)


@runtime_checkable
class LLMProvider(Protocol):
    """Generates chat completions and summaries."""

    def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...

    async def complete(self, messages: list[ChatMessage]) -> str: ...


class OpenAILLMProvider:
    """OpenAI chat-completions implementation of `LLMProvider`."""

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        client = get_openai_client()
        stream = await client.chat.completions.create(
            model=settings.chat_model,
            messages=_as_params(messages),
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage:  # final usage-only chunk
                get_usage_meter().add_chat(
                    chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                )
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def complete(self, messages: list[ChatMessage]) -> str:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=settings.chat_model, messages=_as_params(messages)
        )
        if response.usage:
            get_usage_meter().add_chat(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )
        return (response.choices[0].message.content or "").strip()
