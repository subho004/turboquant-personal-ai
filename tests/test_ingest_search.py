"""Integration tests for the upload -> index -> search/chat flow."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

_DOC = (
    b"# Vector DBs\n\n## Benchmarks\n\n"
    b"TurboVec is faster than FAISS for this benchmark and uses far less RAM.\n"
)


async def _create_folder(client: AsyncClient, name: str = "Notes") -> int:
    resp = await client.post("/api/v1/folders", json={"name": name})
    assert resp.status_code == 201
    return int(resp.json()["data"]["id"])


async def _upload(client: AsyncClient, folder_id: int, content: bytes = _DOC) -> dict:
    resp = await client.post(
        f"/api/v1/folders/{folder_id}/files",
        files={"file": ("vectors.md", content, "text/markdown")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


@pytest.mark.asyncio
async def test_upload_indexes_file(client: AsyncClient) -> None:
    folder_id = await _create_folder(client)
    data = await _upload(client, folder_id)

    assert data["status"] == "indexed"
    assert data["num_chunks"] >= 1


@pytest.mark.asyncio
async def test_duplicate_upload_is_idempotent(client: AsyncClient) -> None:
    folder_id = await _create_folder(client)
    first = await _upload(client, folder_id)
    second = await _upload(client, folder_id)
    assert first["id"] == second["id"]


@pytest.mark.asyncio
async def test_search_finds_indexed_content(client: AsyncClient) -> None:
    folder_id = await _create_folder(client)
    await _upload(client, folder_id)

    resp = await client.post(
        "/api/v1/search/query", json={"query": "TurboVec", "top_k": 5}
    )
    assert resp.status_code == 200
    sources = resp.json()["data"]["sources"]
    assert sources, "hybrid search should match the keyword 'TurboVec'"
    assert any(s["file_name"] == "vectors.md" for s in sources)


@pytest.mark.asyncio
async def test_unsupported_extension_rejected(client: AsyncClient) -> None:
    folder_id = await _create_folder(client)
    resp = await client.post(
        f"/api/v1/folders/{folder_id}/files",
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_streams_grounded_answer(client: AsyncClient) -> None:
    folder_id = await _create_folder(client)
    await _upload(client, folder_id)

    events: list[dict] = []
    async with client.stream(
        "POST", "/api/v1/chat/stream", json={"message": "Is TurboVec fast?"}
    ) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))

    types = [e["type"] for e in events]
    assert "sources" in types and "done" in types
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert "files." in answer  # FakeLLM's grounded answer
    sources_event = next(e for e in events if e["type"] == "sources")
    assert sources_event["sources"], "answer should cite at least one source"
