"""Test the usage/cost meter endpoint shape."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_usage_totals_shape(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/usage/totals")
    assert resp.status_code == 200
    data = resp.json()["data"]
    for key in ("total_tokens", "cost_usd", "requests", "embed_tokens"):
        assert key in data
    assert isinstance(data["cost_usd"], (int, float))
