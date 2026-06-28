"""Pure hashing helpers."""

from __future__ import annotations

import hashlib


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of raw bytes (used for file dedupe)."""

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of text (used for the embedding cache key)."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
