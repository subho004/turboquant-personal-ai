"""Unit tests for the heading-aware chunker."""

from __future__ import annotations

from app.services.chunker import chunk_markdown


def test_captures_heading_path() -> None:
    markdown = "# Title\n\n## Background\n\nTurboVec is fast.\n"
    drafts = chunk_markdown(markdown)

    assert drafts, "expected at least one chunk"
    assert any("Background" in d.heading_path for d in drafts)
    assert all(d.token_count > 0 for d in drafts)


def test_long_section_splits_into_multiple_chunks() -> None:
    body = " ".join(f"word{i}" for i in range(4000))
    drafts = chunk_markdown(f"# Doc\n\n{body}", chunk_tokens=200, overlap=20)

    assert len(drafts) > 1
    ordinals = [d.ordinal for d in drafts]
    assert ordinals == sorted(ordinals)  # ordinals are monotonic


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_markdown("   \n  ") == []
