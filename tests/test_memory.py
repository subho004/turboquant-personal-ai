"""Integration tests for conversation memory and second-brain queries."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

from app.services.vector_store import MEMORY, get_vector_store

_DOC = b"# GNNs\n\nGraph neural networks aggregate messages over edges.\n"


async def _setup(client: AsyncClient) -> int:
    resp = await client.post("/api/v1/folders", json={"name": "Research"})
    folder_id = int(resp.json()["data"]["id"])
    await client.post(
        f"/api/v1/folders/{folder_id}/files",
        files={"file": ("gnn.md", _DOC, "text/markdown")},
    )
    return folder_id


@pytest.mark.asyncio
async def test_chat_writes_a_memory(client: AsyncClient) -> None:
    await _setup(client)
    # Use body words so the hybrid keyword path grounds the answer (the "# GNNs"
    # heading is stored as heading_path, not in the FTS-indexed chunk text).
    async with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"message": "How do graph neural networks aggregate messages?"},
    ) as resp:
        async for _ in resp.aiter_lines():
            pass

    # The grounded exchange should have stored one memory in the memory index.
    assert get_vector_store().count(MEMORY) >= 1


@pytest.mark.asyncio
async def test_which_files_mention(client: AsyncClient) -> None:
    await _setup(client)
    resp = await client.get("/api/v1/search/mentions", params={"q": "graph"})
    assert resp.status_code == 200
    rows = resp.json()["data"]
    assert any(r["file_name"] == "gnn.md" for r in rows)


@pytest.mark.asyncio
async def test_search_history_records_queries(client: AsyncClient) -> None:
    await _setup(client)
    await client.post("/api/v1/search/query", json={"query": "messages over edges"})

    resp = await client.get("/api/v1/search/history")
    queries = [row["query"] for row in resp.json()["data"]]
    assert "messages over edges" in queries


@pytest.mark.asyncio
async def test_conversations_listed_after_chat(client: AsyncClient) -> None:
    await _setup(client)
    async with client.stream(
        "POST", "/api/v1/chat/stream", json={"message": "Explain GNNs"}
    ) as resp:
        events = [
            json.loads(line[6:])
            async for line in resp.aiter_lines()
            if line.startswith("data: ")
        ]
    conversation_id = next(e["conversation_id"] for e in events if e["type"] == "done")

    resp = await client.get("/api/v1/chat/conversations")
    ids = [c["id"] for c in resp.json()["data"]]
    assert conversation_id in ids
