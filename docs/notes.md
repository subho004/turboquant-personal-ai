# TurboQuant Personal AI — Engineering Notes

> A deep, explanatory companion to the codebase: what this project is, the history and
> theory it stands on, how every part fits together (with diagrams), the decisions and
> trade-offs behind them, and where it can go next.
>
> Companion doc: [README.md](README.md) (quick start). This file is the "why and how it really works".

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [History &amp; theoretical background](#2-history--theoretical-background)
3. [System architecture (diagrams)](#3-system-architecture)
4. [The ingestion pipeline](#4-the-ingestion-pipeline)
5. [The retrieval pipeline](#5-the-retrieval-pipeline)
6. [The chat (RAG) loop](#6-the-chat-rag-loop)
7. [Conversation memory](#7-conversation-memory)
8. [How TurboQuant / TurboVec works](#8-how-turboquant--turbovec-works)
9. [Data model](#9-data-model)
10. [The benchmark](#10-the-benchmark)
11. [Evaluation](#11-evaluation)
12. [Security &amp; robustness](#12-security--robustness)
13. [Key design decisions &amp; trade-offs](#13-key-design-decisions--trade-offs)
14. [Known limitations](#14-known-limitations)
15. [Future roadmap](#15-future-roadmap)
16. [References](#16-references)

---

## 1. What this project is

**Personal AI (TurboVec)** is a local-first "**ChatGPT with a perfect memory of a folder**."
Instead of wiring up connectors to Gmail, Slack, or Drive, you drop files into folders and
chat over them. Every file is parsed, chunked, embedded, and indexed so the assistant can
answer questions grounded in _your_ documents, with clickable citations back to the source.

The defining choice is the retrieval engine: **[TurboVec](https://github.com/RyanCodrai/turbovec)**,
a Rust vector index (with Python bindings) implementing Google Research's **TurboQuant**
quantization. It compresses embeddings to 2–4 bits per dimension and searches them with
hand-written SIMD kernels — fitting a 10-million-document corpus that would take 31 GB as
float32 into roughly 4 GB, while searching as fast as or faster than FAISS.

In one sentence: **the LLM does the reasoning; TurboVec does the remembering.**

### What it can do today

- Create folders, upload files (pdf, docx, pptx, xlsx, csv, md, txt, html, images, audio).
- Automatic parse → chunk → embed → index on upload.
- **Hybrid retrieval** (semantic + keyword) with **LLM reranking** and **answer grounding**.
- **Streaming chat** with inline `[S1]`-style citations and a source preview.
- **Follow-up awareness** (multi-turn questions are rewritten into standalone queries).
- **Cross-conversation memory** (a second vector index of summarized exchanges).
- **Second-brain queries**: "summarise everything about X", "which files mention Y?", search history.
- A live **TurboVec-vs-FAISS benchmark dashboard** and a **token/cost meter**.

---

## 2. History & theoretical background

This project sits on top of three decades of ideas. Understanding them explains why the
design looks the way it does.

### 2.1 The vector-search problem

Semantic search represents text as high-dimensional vectors (embeddings) and finds the
"nearest" vectors to a query. The naive approach — store every vector as float32 and compare
the query against all of them ("flat" / exact search) — is accurate but expensive: a
1536-dimensional vector is 6,144 bytes, so 10M vectors need ~31 GB of RAM, and every search
touches all of them. Two families of techniques attack this:

- **Approximate Nearest Neighbor (ANN) indexes** (HNSW graphs, IVF inverted lists) reduce how
  _many_ vectors you compare.
- **Vector quantization** reduces how _big_ each vector is.

### 2.2 Vector quantization & FAISS

**Product Quantization (PQ)**, introduced by Jégou, Douze & Schmid (2011), splits each vector
into sub-vectors and replaces each with the id of the nearest centroid in a learned codebook —
compressing 16× or more. Facebook AI's **[FAISS](https://github.com/facebookresearch/faiss)**
(2017) made PQ and its variants (IVFPQ, PQ-FastScan with SIMD) the industry default. The
catch: PQ **must be trained** on a representative sample to build its codebooks, and adding
data that drifts from that distribution degrades recall until you retrain.

### 2.3 TurboQuant (Google Research, 2025–2026)

**TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874); ICLR 2026 —
[OpenReview](https://openreview.net/forum?id=tO3ASKZlok); authors Zandieh, Daliri, Hadian,
Mirrokni) is a **data-oblivious, online** quantization method with provably **near-optimal
distortion** across all bit-widths and dimensions. "Data-oblivious" is the key word: it needs
**no training phase and no codebook**, so vectors can be added at any time without rebuilds.
It rotates each vector so coordinates follow a predictable distribution, applies optimal
scalar (Lloyd-Max) quantization, and corrects inner-product bias with a 1-bit residual pass
(QJL). Google originally framed it for compressing the **KV cache** during LLM inference
(3-bit, ~6× memory reduction, up to ~8× faster attention on H100s) — but the same math is a
drop-in win for embedding search. See [§8](#8-how-turboquant--turbovec-works) for the details.

### 2.4 TurboVec

**[TurboVec](https://github.com/RyanCodrai/turbovec)** is an open-source (MIT) Rust
implementation of TurboQuant as a vector index, with Python bindings and hand-written
**NEON (ARM)** and **AVX-512 (x86)** SIMD search kernels. It exposes `TurboQuantIndex`
(append-only) and `IdMapIndex` (stable external ids, O(1) delete, allowlist filtering).
This project uses `IdMapIndex` — see [§13](#13-key-design-decisions--trade-offs).

### 2.5 Hybrid search & RRF

Pure vector search misses exact terms (error codes, names, rare tokens); pure keyword search
misses paraphrases. **Hybrid search** runs both and fuses them. The fusion method here is
**Reciprocal Rank Fusion (RRF)**, introduced by Cormack, Clarke & Büttcher in their 2009 SIGIR
paper _"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"_.
RRF ignores raw scores (which live on incompatible scales — cosine is [-1,1], BM25 is unbounded)
and sums `1/(k + rank)` across lists. It is training-free and robust — exactly what we want.

### 2.6 RAG, MarkItDown, and this project's genesis

**Retrieval-Augmented Generation (RAG)** — retrieve relevant context, then let an LLM answer
from it — is the dominant pattern for grounding LLMs in private data. To feed it, raw files
must become clean text; we use Microsoft's **[MarkItDown](https://github.com/microsoft/markitdown)**
(pdf/docx/pptx/xlsx/images-OCR → markdown) and OpenAI **Whisper** for audio.

The project itself began as the **"TurboQuant Personal AI MVP"** brief: build a retrieval-and-
memory-focused assistant over a local folder, simulate a realistic knowledge base, and prove
TurboVec's advantages against FAISS. This was then implemented phase by phase into the system
documented here. Notably, the requested chat model (`gpt-5.5-mini`) is not a released id, so the
code defaults to `gpt-5-mini` and keeps every model name configurable via environment variables.

---

## 3. System architecture

The system is a FastAPI backend with a clean **layered** structure (routes → services →
repositories → storage) and a dependency-light single-page web UI. Two storage engines split
responsibility: **SQLite** owns all metadata and chunk text; **TurboVec** owns only the
compressed vectors, keyed by `chunk_id`.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Browser — single-page UI (web/index.html, app.js, benchmark.html)           │
│  Folders · drag-upload · streaming chat · source chips · history · cost meter│
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  REST + Server-Sent Events  (fetch / streaming)
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  FastAPI app (main.py → app/api/v1/*)                                         │
│  Routes: folders · files · chat · search · benchmark · usage · health         │
│  - parse/validate input, call a service, wrap response (utils/response.py)     │
│  - get_session owns the transaction (commit on success / rollback on error)    │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  dependency injection (app/api/deps.py)
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  Services (app/services/*) — business logic / orchestration                    │
│                                                                                │
│   Ingestion            Retrieval                  Generation      Meta         │
│   ┌──────────┐         ┌────────────────────┐     ┌──────────┐   ┌─────────┐  │
│   │ Parser   │         │ RetrievalService   │     │ LLM      │   │ Usage   │  │
│   │ Chunker  │         │  hybrid + RRF      │────▶│ stream   │   │ Meter   │  │
│   │ Embedder │────┐    │  + Reranker        │     │ complete │   │ Bench-  │  │
│   │ Ingest   │    │    │  + memory recall   │     └──────────┘   │ mark    │  │
│   └────┬─────┘    │    └─────────┬──────────┘                    └─────────┘  │
└────────┼──────────┼──────────────┼────────────────────────────────────────────┘
         │          │              │
  Repositories (app/repositories/*) — the ONLY layer that touches the DB
         │          │              │
┌────────▼────┐  ┌──▼───────────┐  ▼
│  SQLite     │  │ TurboVec MAIN│  TurboVec MEMORY
│ (metadata + │  │ index .tvim  │  index .tvim
│ chunk text  │  │ chunk vectors│  conversation-summary vectors
│ + FTS5)     │  └──────────────┘
└─────────────┘     (data/index/*.tvim, data/app.db, data/uploads/)
```

### Layer responsibilities (enforced by `.github/copilot-instructions.md`)

| Layer        | Directory                            | May do                                            | Must not do                 |
| ------------ | ------------------------------------ | ------------------------------------------------- | --------------------------- |
| Routes       | [app/api/v1](app/api/v1)             | validate input, call a service, wrap the response | DB queries, business logic  |
| Services     | [app/services](app/services)         | orchestrate use-cases, call repos & providers     | touch HTTP types, build SQL |
| Repositories | [app/repositories](app/repositories) | all ORM/DB access                                 | business rules              |
| Schemas      | [app/schemas](app/schemas)           | Pydantic I/O DTOs                                 | import DB models            |
| Utils        | [utils](utils)                       | pure stateless helpers                            | hold state, DB, HTTP        |

Cross-cutting infrastructure lives in [app/core](app/core): typed settings
([config.py](app/core/config.py)), the JSON logger ([logging.py](app/core/logging.py)), and the
`AppError` exception hierarchy ([exceptions.py](app/core/exceptions.py)) that handlers convert
into a uniform response shape.

---

## 4. The ingestion pipeline

Uploading a file runs it through a synchronous pipeline that keeps SQLite, the FTS5 mirror, and
the TurboVec index consistent. Orchestrated by [IngestService](app/services/ingest.py), invoked
from [FileService.upload](app/services/file_service.py).

```
 Upload (multipart)                                       per-file status
      │                                                   ─────────────────
      ▼                                                   pending
 1. Validate: extension allowlist, ≤25 MB, non-empty        │
      │  (path-traversal-safe filename)                      │
      ▼                                                       ▼
 2. sha256(content) ──► already in this folder? ─yes─► return existing (idempotent)
      │ no                                                    │
      ▼                                                       ▼
 3. Write bytes to data/uploads/<folder>/<hash>_<name>     parsing
      │                                                       │
      ▼                                                       │
 4. Parse → markdown   (MarkItDown, or Whisper for audio)     │
      │                                                       ▼
 5. Chunk (heading-aware, ~500 tok, 80 overlap)            embedding
      │                                                       │
      ▼                                                       │
 6. Embed all chunks  (OpenAI, batched ≤100, hash-cached)     │
      │                                                       │
      ▼                                                       ▼
 7. Persist: chunk rows + FTS5 mirror (SQLite)             indexed
      │      + vectors → TurboVec MAIN index
      ▼
   mark_indexed(parsed_chars, num_chunks)   ── on any error ──► status=error, reason stored
```

Design notes:

- **Chunking** ([chunker.py](app/services/chunker.py)) splits on markdown headings first — so a
  chunk carries a `heading_path` like `"TurboVec > RAM"` for precise citations — then slices each
  section into overlapping token windows using `tiktoken`'s `cl100k_base` tokenizer.
- **Embedding** ([embeddings.py](app/services/embeddings.py)) batches up to 100 inputs per request
  and caches by content hash, so re-uploads and rebuilds never re-pay for identical text. It also
  reports token usage to the [UsageMeter](app/services/usage.py).
- **Re-indexing** (`IngestService.reindex`) removes a file's old `chunk_id`s from the index and
  DB, then re-runs — which is only possible because `IdMapIndex` supports O(1) delete by id.
- **Synchronous on purpose**: ingestion runs inside the upload request so the HTTP response
  reflects the final `indexed`/`error` status. This trades large-upload latency for simplicity and
  crash-evidence (see [§13](#13-key-design-decisions--trade-offs)).

---

## 5. The retrieval pipeline

[RetrievalService.retrieve](app/services/retrieval.py) is the heart of search. It fuses two
signals, reranks, and pulls in memory — all behind one method used by both chat and search.

```
 query ─► embed query (OpenAI)
   │
   ├─────────────► TurboVec MAIN search (k=30)            ─┐  ranked id list A
   │               (allowlist = folder subtree, optional)  │
   │                                                        ├─► RRF fuse
   └─────────────► SQLite FTS5 BM25 keyword search (k=30)  ─┘   score = Σ 1/(60 + rank)
                                                                  │
                                          top-30 candidate ids ◄──┘
                                                  │
                                   fetch chunk text + file name (1 joined query)
                                                  │
                                   ┌──────────────▼──────────────┐
                                   │ Reranker (LLM)              │  reorder candidates,
                                   │  rerank_enabled? candidates>k│  keep best top_k=12
                                   │  → graceful fallback to RRF  │
                                   └──────────────┬──────────────┘
                                                  │
                              + memory recall (TurboVec MEMORY, k=6)
                                                  │
                                                  ▼
                       RetrievalResult{ chunks, memories, best_vector_score, has_keyword_hit }
```

Why each piece exists:

- **Hybrid + RRF** ([retrieval.py](app/services/retrieval.py), `_rrf`, `_RRF_K = 60`): vectors catch
  meaning, BM25 catches exact terms. RRF combines their _ranks_, sidestepping incompatible score
  scales (Cormack et al., 2009).
- **Folder scoping**: an optional **allowlist** of `chunk_id`s (the folder subtree) is passed into
  TurboVec's SIMD kernel, which filters at search time. The store intersects the allowlist with its
  known-id set so a stale id can never raise `KeyError`.
- **Reranking** ([reranker.py](app/services/reranker.py)): retrieve wide (30), then have the LLM
  reorder to the final 12. If the model's output can't be parsed, it falls back to RRF order — so
  reranking only ever helps. Gated by `settings.rerank_enabled`.
- **`has_keyword_hit` / `best_vector_score`**: carried out so the chat layer can decide whether the
  answer is _grounded_ (a real match) versus the vector index merely returning its k nearest
  neighbours for an irrelevant query.

---

## 6. The chat (RAG) loop

[ChatService.answer_stream](app/services/chat_service.py) turns a message into a streamed,
grounded, cited answer and persists everything. It yields transport-agnostic event dicts; the
[chat route](app/api/v1/chat.py) adapts them to Server-Sent Events.

```
 sequence for POST /api/v1/chat/stream
 ─────────────────────────────────────
 Client            ChatService                 Retrieval/LLM            SQLite / TurboVec
   │  message + conv_id?  │                          │                        │
   │─────────────────────►│ get/create conversation  │                        │
   │                      │ list prior messages ─────┼───────────────────────►│
   │                      │ rewrite follow-up query ─►│ LLM.complete           │
   │                      │ store user message ──────┼───────────────────────►│
   │                      │ retrieve(query) ─────────►│ hybrid+rerank+memory   │
   │  event: start        │◄─────────────────────────│                        │
   │◄─────────────────────│                          │                        │
   │  event: sources      │  build [S1..] context     │                        │
   │◄─────────────────────│                          │                        │
   │                      │ grounded? ── no ─► canned "I don't have that…"     │
   │                      │            yes            │                        │
   │  event: token (×N) ◄─┼──── stream answer ───────►│ LLM.stream_chat        │
   │◄─────────────────────│                          │                        │
   │                      │ store assistant msg ─────┼───────────────────────►│
   │                      │ record search history ───┼───────────────────────►│
   │  event: done         │                          │                        │
   │◄─────────────────────│                          │                        │
   │                      │ summarise → memory index ┼───────────────────────►│ (best-effort)
   │                      │ session.commit()         │                        │
```

The **system prompt** instructs the model to answer only from the numbered sources/memories,
cite inline as `[S1]`, admit when context lacks the answer, and **treat all retrieved text as
untrusted data, never as instructions** (prompt-injection defense). **Grounding** logic:
the answer is generated only if `best_vector_score ≥ min_score` **or** there was an exact keyword
hit **or** a relevant memory exists; otherwise it returns a canned "not in your files" message.
This matters because vector search _always_ returns k neighbours, even for nonsense queries.

---

## 7. Conversation memory

The assistant remembers across conversations using a **second, separate** TurboVec index
([MemoryService](app/services/memory_service.py)). After each grounded exchange it asks the LLM
for a one-sentence summary, embeds it, and stores it with the `Memory` row id.

```
 exchange ─► LLM summary ("TurboVec saves ~8× RAM vs FAISS") ─► embed ─► MEMORY index
                                                                            │
 later query ─► embed ─► search MEMORY (k=6) ──────────────────────────────►│
                         merged into RetrievalResult.memories ─► fed to LLM alongside docs
```

Because MAIN (chunks) and MEMORY (summaries) are **distinct indexes**, ids never collide — a
memory simply reuses its SQLite primary key as its vector id, no global offset needed. This
realizes the brief's "Remember that X… → What did we conclude about X?" flow.

---

## 8. How TurboQuant / TurboVec works

TurboQuant's goal: store a unit vector in a few bits per dimension while keeping inner-product
(similarity) estimates accurate and **unbiased**, with **no training**. The pipeline TurboVec
implements, step by step:

```
 raw embedding  ─►  1. NORMALIZE to a unit direction on the hypersphere
                ─►  2. RANDOM ROTATION  (a fixed random rotation makes every coordinate
                                          follow a predictable Beta-like distribution —
                                          this is the "data-oblivious" trick)
                ─►  3. PER-COORDINATE CALIBRATION (TQ+)  shift/scale per dimension
                ─►  4. LLOYD-MAX SCALAR QUANTIZATION  (optimal buckets:
                                          4 levels for 2-bit, 16 for 4-bit)
                ─►  5. BIT-PACK  (1536 dims × 4 bits = 768 bytes; ×2 bits = 384 bytes)
                ─►  6. STORE per-vector renormalization scalar
 search:  rotate the query the same way, score against packed codes with SIMD,
          apply length-renormalized correction (+ QJL 1-bit residual removes IP bias)
```

Why it's fast and small:

- **Compression**: a 1536-dim FP32 vector (6,144 B) → **384 B at 2-bit (16×)** or **768 B at
  4-bit (8×)**. A 10M-doc corpus drops from **31 GB → ~4 GB**.
- **Speed**: scoring runs over bit-packed codes with hand-written **NEON/AVX-512** kernels, so it
  is memory-bandwidth-bound on tiny data — beating FAISS FastScan by **10–19% on Apple Silicon**.
- **Online**: because the rotation is data-oblivious, there is **no codebook to train**; `add()`
  indexes immediately. (Trade-off: TurboVec's TQ+ calibration freezes after the first batch — see
  the warm-start note in [§14](#14-known-limitations).)
- **Quality**: near-optimal distortion is _proven_ (arXiv:2504.19874); empirically TurboVec beats
  FAISS IndexPQ by **0.2–1.9 points at Recall@1**.

We use 4-bit `IdMapIndex` (`dim=1536`) — the best recall/speed balance and the only variant with
stable ids, O(1) delete, and allowlist filtering ([vector_store.py](app/services/vector_store.py)).

---

## 9. Data model

SQLite is the source of truth for everything except the vectors. ORM models live in
[app/models](app/models).

```
 folders ──┐ (self-referential parent_id → folder subtree)
           │
           └──< files >── status: pending│parsing│embedding│indexed│error
                  │        content_hash (dedupe), rel_path, num_chunks
                  │
                  └──< chunks >  id == TurboVec MAIN vector id
                          │       text, token_count, heading_path, ordinal
                          │       └── mirrored into chunks_fts (FTS5, BM25)
                          │
 conversations ──< messages >  role, content, cited_chunk_ids[]
        │
        └──< memories >  id == TurboVec MEMORY vector id; kind, text

 search_history  query, top_chunk_ids[], answer_preview   (every search becomes memory)
```

Key idea: **`chunks.id` and `memories.id` double as the `uint64` external ids in their respective
TurboVec indexes.** A single id always resolves to exactly one row, and the `.tvim` files can be
rebuilt from SQLite if needed. The `chunks_fts` virtual table is a content-synced FTS5 mirror that
powers the BM25 half of hybrid search ([chunk_repository.py](app/repositories/chunk_repository.py)).

---

## 10. The benchmark

[BenchmarkService](app/services/benchmark_service.py) backs the "why TurboVec" claim with live
numbers on a synthetic corpus (so size/latency/build/recall can be measured without huge OpenAI
bills). It compares **TurboVec (4-bit)** against **FAISS IndexFlat** (exact, the recall ground
truth) and **FAISS IndexPQ** (the compressed competitor).

**Method**: generate N normalized random 1536-dim vectors and 200 queries; build each index;
measure build time, p50/p95 search latency over the query set, on-disk index size, and Recall@10
versus the exact FlatIP results. CPU-bound work runs in a worker thread.

**Observed (5,000 vectors, this machine)** — illustrative, from a live run:

| Index                | Build (s) | p50 (ms) | Size (MB) | Compression | Recall@10 | Training |
| -------------------- | --------- | -------- | --------- | ----------- | --------- | -------- |
| **TurboVec (4-bit)** | 0.20      | **0.26** | 3.7       | 7.9×        | **0.86**  | none     |
| FAISS IndexPQ        | 1.18      | 5.31     | 2.0       | 14.9×       | 0.16      | required |
| FAISS IndexFlat      | —         | 0.42     | 29.3      | 1×          | 1.00      | none     |

The headline: TurboVec searches **~20× faster than FAISS PQ here, builds with no training step,
and holds far higher recall on this data** (PQ's learned codebooks struggle on unstructured random
vectors — an honest caveat surfaced in the result's `note`). On real embeddings the gaps narrow,
but the _no-training, online-insert, low-RAM_ story holds. The dashboard
([web/benchmark.html](web/benchmark.html)) renders these as bar charts plus a table.

---

## 11. Evaluation

Beyond "it looks right," [scripts/evaluate.py](scripts/evaluate.py) measures retrieval quality
honestly. It ingests a small golden corpus through the **real** pipeline (MarkItDown → OpenAI
embeddings → TurboVec hybrid retrieval) and scores a hand-labelled question set:

- **hit@k** — did the expected file appear in the top-k sources?
- **MRR** — mean reciprocal rank of the expected file.

On the bundled golden set it achieves **100% hit@5 and MRR 1.0**, validating that the end-to-end
real pipeline retrieves the right documents. The automated test suite ([tests](tests)) covers the
chunker, vector store (add/search/remove/allowlist/idempotency/index-separation), and the full
upload→search→chat→memory flow using **offline fakes** for OpenAI, so it runs free and
deterministically (19 tests, green; ruff + mypy clean).

---

## 12. Security & robustness

- **Uploads are an attack surface**: extension allowlist + 25 MB cap + path-traversal-safe names
  (`Path(name).name` strips directories) in [file_service.py](app/services/file_service.py).
- **Prompt injection**: retrieved document text can contain "ignore previous instructions." The
  system prompt explicitly frames all sources as untrusted data and forbids treating them as
  commands; retrieved text never triggers tools.
- **Secrets** stay server-side: `OPENAI_API_KEY` lives only in the backend; the UI never sees it.
- **Durability**: each `.tvim` is written to a temp file then atomically renamed; a single
  `asyncio.Lock` serializes all index writes (the index is an in-memory structure). On any ingest
  error the file is marked `error` with the reason, surfaced in the UI — never a silent drop.
- **Transaction safety**: `get_session` commits on success / rolls back on error, so a failed
  request leaves no partial rows.

---

## 13. Key design decisions & trade-offs

| Decision                                                | Why                                                                                 | Trade-off accepted                            |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------- |
| **`IdMapIndex` over `TurboQuantIndex`**                 | Needs stable ids, O(1) delete, allowlist filtering for re-indexing + folder scoping | Slightly larger than the append-only variant  |
| **Two separate indexes** (MAIN, MEMORY)                 | Disjoint id spaces, independent lifecycles; a uint64 always maps to one source      | Two files to persist/load                     |
| **SQLite as source of truth, vectors-only in TurboVec** | Cheap metadata + chunk text; index is rebuildable; FTS5 comes free                  | Must keep the two stores consistent           |
| **Hybrid + RRF + rerank**                               | Best answer quality: meaning + exact terms + LLM judgment                           | Extra LLM call per query (gated/configurable) |
| **Synchronous ingestion**                               | Response reflects final status; simplest correct path                               | Large uploads block the request               |
| **Streaming via SSE**                                   | "Feels like ChatGPT"; simple over HTTP                                              | Manual event framing                          |
| **Follow-up rewriting only with history**               | Avoids corrupting first-turn queries; keeps tests deterministic                     | An LLM call on multi-turn                     |
| **`gpt-5-mini` default**                                | `gpt-5.5-mini` is unreleased; all model names are env-configurable                  | Swap when 5.5-mini ships                      |
| **Single writer lock**                                  | TurboVec index is in-memory; concurrent writes would corrupt                        | Writes serialize (fine at this scale)         |

---

## 14. Known limitations

- **TQ+ calibration freezes after the first batch.** TurboVec calibrates on the first vectors
  added; an unrepresentative first batch can hurt recall. Mitigation (planned): warm-start the
  index with a representative sample before live ingestion.
- **Synchronous ingestion** blocks the upload request for large files; there is no progress stream
  during indexing yet (the UI shows a final status).
- **Memory ↔ index drift** is possible if the process crashes between a SQLite commit and the
  debounced index write; a startup reconciliation job is planned but not yet built.
- **No auth / multi-tenant** — single-user, local-first by design.
- **Benchmark recall uses synthetic random vectors**, which understates FAISS PQ (its codebooks
  need structure); the result `note` says so. A "real embeddings" mode would give honest recall.
- **Reranking & rewriting add latency/cost** per turn (configurable, but on by default).

---

## 15. Future roadmap

**Near term (polish & quality)**

- Warm-start calibration; **small-to-big** chunking (embed small, feed the parent section to the LLM).
- Background/async ingestion with a live SSE status stream; a `watchdog` folder watcher for
  auto-reindex on file change (the original "Ctrl+S → re-embed" vision).
- More second-brain endpoints already half-built: date-range "daily notes" queries
  (`ChunkRepository.ids_by_date_range` exists), "find similar files," "how are these connected."

**Mid term (durability & scale)**

- Startup **reconciliation** of SQLite ↔ index drift; cache raw vectors as BLOBs for instant
  rebuilds without re-embedding.
- Benchmark on **real embeddings** at 100k–500k scale with persisted run history and trend charts;
  add a true RAM measurement (psutil) alongside on-disk size.
- Deeper **evaluation**: LLM-judge faithfulness + citation-correctness scoring, wired into CI to
  prove reranking beats naive retrieval.

**Long term (the "second brain" OS)**

- Knowledge-graph view of file relationships; "what have I forgotten about X?" (surface topics
  not recently opened/cited); duplicate-note detection.
- Pluggable embedding/LLM providers (local models via the same `Protocol` interfaces).
- 2-bit index option and a UI toggle to compare 2-bit vs 4-bit recall/size live.

---

## 16. References

**TurboQuant & TurboVec**

- TurboQuant: _Online Vector Quantization with Near-optimal Distortion Rate_ — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) · [OpenReview (ICLR 2026)](https://openreview.net/forum?id=tO3ASKZlok)
- Google Research blog — [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- TurboVec library — [github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) · [README](https://github.com/RyanCodrai/turbovec/blob/main/README.md)
- Explainer — [TurboVec: Rust and Python Vector Search (Neel Shah)](https://neelshah18.com/blog/turbovec-rust-python-vector-index/)

**Retrieval & fusion**

- Cormack, Clarke & Büttcher (2009), _Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods_, SIGIR — [ResearchGate](https://www.researchgate.net/publication/221301121_Reciprocal_Rank_Fusion_outperforms_Condorcet_and_Individual_Rank_Learning_Methods)
- FAISS — [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- RRF for hybrid search (overview) — [glaforge.dev](https://glaforge.dev/posts/2026/02/10/advanced-rag-understanding-reciprocal-rank-fusion-in-hybrid-search/)

**Parsing, embeddings, models**

- MarkItDown — [github.com/microsoft/markitdown](https://github.com/microsoft/markitdown) · [Real Python guide](https://realpython.com/python-markitdown/)
- OpenAI embeddings — [Vector embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) · [New embedding models](https://openai.com/index/new-embedding-models-and-api-updates/)

**This project**

- [README.md](README.md) — quick start and API surface
- [.github/copilot-instructions.md](.github/copilot-instructions.md) — engineering guide governing the codebase
