"""Heading-aware, token-windowed markdown chunker.

Splits text on markdown headings first (so a chunk's ``heading_path``
can be cited), then slices each section into overlapping token windows.
Pure and stateless — no I/O, no external services.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from app.core.config import settings

# text-embedding-3-* and the GPT models all use the cl100k/o200k families;
# cl100k_base is a safe, dependency-free tokenizer for length budgeting.
_ENCODING = tiktoken.get_encoding("cl100k_base")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before it is persisted (no id yet)."""

    ordinal: int
    text: str
    token_count: int
    heading_path: str


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading_path, body)`` sections by headings."""

    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buffer: list[str] = []
    heading_path = ""

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((heading_path, body))

    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            heading_path = " > ".join(stack)
        else:
            buffer.append(line)

    flush()
    if not sections:  # no headings at all
        body = markdown.strip()
        return [("", body)] if body else []
    return sections


def _window(tokens: list[int], size: int, overlap: int) -> list[list[int]]:
    """Slice a token list into overlapping windows."""

    if len(tokens) <= size:
        return [tokens]
    step = max(1, size - overlap)
    return [
        tokens[i : i + size]
        for i in range(0, len(tokens), step)
        if tokens[i : i + size]
    ]


def chunk_markdown(
    markdown: str,
    chunk_tokens: int | None = None,
    overlap: int | None = None,
) -> list[ChunkDraft]:
    """Chunk markdown into overlapping, heading-tagged token windows."""

    size = chunk_tokens or settings.chunk_tokens
    over = overlap if overlap is not None else settings.chunk_overlap

    drafts: list[ChunkDraft] = []
    ordinal = 0
    for heading_path, body in _split_sections(markdown):
        tokens = _ENCODING.encode(body)
        for window in _window(tokens, size, over):
            text = _ENCODING.decode(window).strip()
            if not text:
                continue
            drafts.append(
                ChunkDraft(
                    ordinal=ordinal,
                    text=text,
                    token_count=len(window),
                    heading_path=heading_path,
                )
            )
            ordinal += 1
    return drafts
