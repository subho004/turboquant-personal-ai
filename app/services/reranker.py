"""LLM reranker.

Takes the hybrid-retrieval candidate set and asks the LLM to order the
passages by relevance to the query, returning the best ``top_n``. Falls
back to the original (RRF) order if the model output can't be parsed, so
reranking never breaks retrieval.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.services.llm import LLMProvider
from app.services.retrieval import RetrievedChunk

logger = get_logger(__name__)

_SNIPPET_CHARS = 280
_SYSTEM = (
    "You are a search reranker. Given a query and numbered passages, return the "
    "passage numbers ordered most-relevant-first for answering the query. Reply "
    "with ONLY a comma-separated list of numbers, best first, no other text."
)


class Reranker:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if len(candidates) <= top_n:
            return candidates

        passages = "\n".join(
            f"{i + 1}. {c.file_name}: {c.text[:_SNIPPET_CHARS]}"
            for i, c in enumerate(candidates)
        )
        prompt = f"Query: {query}\n\nPassages:\n{passages}"
        try:
            reply = await self._llm.complete(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
            )
            order = self._parse_order(reply, len(candidates))
            if not order:
                return candidates[:top_n]
            reranked = [candidates[i] for i in order]
            # Append any candidates the model omitted, preserving RRF order.
            seen = set(order)
            reranked += [c for i, c in enumerate(candidates) if i not in seen]
            return reranked[:top_n]
        except Exception:  # noqa: BLE001 — reranking is best-effort
            logger.exception("Rerank failed; falling back to fusion order")
            return candidates[:top_n]

    @staticmethod
    def _parse_order(reply: str, count: int) -> list[int]:
        """Parse '3, 1, 2' -> [2, 0, 1] (0-based, valid, deduped, in order)."""

        seen: set[int] = set()
        order: list[int] = []
        for token in re.findall(r"\d+", reply):
            idx = int(token) - 1
            if 0 <= idx < count and idx not in seen:
                seen.add(idx)
                order.append(idx)
        return order
