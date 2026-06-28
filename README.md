# TurboQuant Personal AI

> **"ChatGPT with a perfect memory of a folder."**
> Upload documents into folders and chat over them with grounded, cited answers —
> semantic search powered by [TurboVec](https://github.com/RyanCodrai/turbovec),
> answers by GPT, parsing by MarkItDown, audio by Whisper.

Built on **Python 3.14.2** + [`uv`](https://docs.astral.sh/uv/) · FastAPI · SQLAlchemy (async SQLite) · TurboVec · FAISS · OpenAI.


## Features

- **Folders + upload** — drag files into a folder; they're parsed → chunked → embedded → indexed automatically.
- **Parse anything** — pdf, docx, pptx, xlsx, csv, md, txt, html, images (OCR) via MarkItDown; audio via Whisper.
- **Hybrid retrieval + reranking** — TurboVec vector search **fused with SQLite FTS5 BM25** (RRF), then an LLM reranks candidates down to the final top-k.
- **Follow-up aware** — multi-turn questions ("how much RAM does *it* save?") are rewritten into standalone search queries using conversation history.
- **Streaming RAG chat** — grounded answers with clickable `[S1]` source citations; refuses when nothing relevant is indexed.
- **Conversation memory + history** — each exchange is summarised into a second TurboVec index and recalled across chats; past chats are listed and reloadable in the UI.
- **Second-brain queries** — "summarise everything about X", "which files mention Y?", search history.
- **Token & cost meter** — live running spend (embeddings + chat) shown in the UI via `/api/v1/usage/totals`.
- **Benchmark dashboard** — live TurboVec vs FAISS on a synthetic corpus: index size, latency, recall, training cost.

## Setup

```bash
uv venv --python 3.14.2
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.sample .env          # then set OPENAI_API_KEY
```

> `gpt-5.5-mini` is not a released model id; the default `CHAT_MODEL` is `gpt-5-mini`.
> Override any model via `.env` (`CHAT_MODEL`, `EMBED_MODEL`, `TRANSCRIBE_MODEL`).

## Run

```bash
uv run uvicorn main:app --reload
```

- **Web UI:** http://127.0.0.1:8000/
- **Benchmark:** http://127.0.0.1:8000/benchmark.html
- **Health:** http://127.0.0.1:8000/health
- **API docs:** http://127.0.0.1:8000/docs

## Test & evaluate

```bash
uv run pytest -v                       # unit + integration (offline, fake providers)
python scripts/evaluate.py             # retrieval hit@k / MRR on a golden set (uses live embeddings)
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/folders` | Create a folder |
| `POST` | `/api/v1/folders/{id}/files` | Upload + index a file |
| `GET`  | `/api/v1/files/{id}/preview` | Extracted text preview |
| `POST` | `/api/v1/chat/stream` | Streaming RAG chat (SSE) |
| `POST` | `/api/v1/search/query` | Hybrid semantic search |
| `POST` | `/api/v1/search/summarise` | Topic / project summary |
| `GET`  | `/api/v1/search/mentions` | Which files mention a term |
| `POST` | `/api/v1/benchmark/run` | TurboVec vs FAISS benchmark |

## Architecture

`routes → services → repositories → (SQLite metadata + TurboVec indexes)`. The OpenAI
embedder/LLM and the MarkItDown parser are injected via `Depends`, so tests swap in fakes.
Chunk text + metadata live in SQLite (with an FTS5 mirror); only compressed vectors live in
TurboVec, keyed by `chunk_id`. See the engineering guide in `.github/copilot-instructions.md`.

## Docker

```bash
docker compose up --build
```
