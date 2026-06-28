"""File parsing — convert any supported file into markdown text.

Office/PDF/text/image files go through Microsoft MarkItDown; audio is
transcribed with the OpenAI transcription API (Whisper).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from markitdown import MarkItDown

from app.core.config import settings
from app.core.exceptions import UnprocessableEntityError
from app.core.logging import get_logger
from app.services.openai_client import get_openai_client

logger = get_logger(__name__)

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".mp4", ".mpeg", ".mpga", ".webm"}
# Everything MarkItDown handles out of the box that we expose in the UI.
DOC_EXTS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".png",
    ".jpg",
    ".jpeg",
}
SUPPORTED_EXTS = AUDIO_EXTS | DOC_EXTS


class Parser:
    """Converts a file on disk into markdown text."""

    def __init__(self) -> None:
        self._md = MarkItDown()

    async def parse(self, path: Path, ext: str) -> str:
        """Return markdown text for ``path``; raise on unsupported/empty."""

        ext = ext.lower()
        if ext in AUDIO_EXTS:
            return await self._transcribe(path)
        if ext in DOC_EXTS:
            return await asyncio.to_thread(self._convert, path)
        raise UnprocessableEntityError(f"Unsupported file type '{ext}'")

    def _convert(self, path: Path) -> str:
        result = self._md.convert(str(path))
        text = (result.text_content or "").strip()
        if not text:
            raise UnprocessableEntityError("No extractable text found in file")
        return text

    async def _transcribe(self, path: Path) -> str:
        client = get_openai_client()
        with path.open("rb") as handle:
            response = await client.audio.transcriptions.create(
                model=settings.transcribe_model, file=handle
            )
        text = (response.text or "").strip()
        if not text:
            raise UnprocessableEntityError("Transcription returned no text")
        return text
