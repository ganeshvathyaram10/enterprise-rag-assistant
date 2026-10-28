# Enterprise Document RAG Assistant

Question answering over an internal document knowledge base with **cited answers**, **hybrid retrieval + cross-encoder reranking**, **abstention when evidence is weak**, and a built-in **evaluation harness** (retrieval quality, answer relevance, faithfulness/hallucination).

Stack: Python · LangChain (text splitters / documents) · Hugging Face (sentence-transformers, cross-encoders, transformers) · ChromaDB · FastAPI · Docker

## Architecture

```
docs (.md/.txt/.pdf) -> chunk (RecursiveCharacterTextSplitter) -> embed (bge-small) -> Chroma (cosine)
                                                                          |
question -> dense top-20 (Chroma) + BM25 top-20 -> Reciprocal Rank Fusion -> cross-encoder rerank -> top-4
         -> abstain if best rerank score < threshold
         -> LLM (flan-t5 local, or HF Inference API) with citation-forcing prompt -> answer + sources
```

| Component | Choice | Why |
|---|---|---|
| Chunking | 600 chars, 100 overlap, paragraph-aware | keeps policy clauses intact |
| Embeddings | `BAAI/bge-small-en-v1.5` | strong quality/size ratio, CPU friendly |
| Retrieval | dense + BM25 fused with RRF | dense misses exact terms ("$750", "CFO"); BM25 misses paraphrase |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | re-scores top-20 candidates jointly with the query |
| Generator | `Qwen2.5-1.5B-Instruct` (local) or HF Inference API | swap via `GENERATOR_BACKEND` |
| Abstention | reranker score threshold | refuses instead of guessing |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.api:app --reload          # indexes data/docs on first boot
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How much is the home office stipend?"}' | jq
```

Docker:

```bash
docker compose up --build
```

Bring your own documents: drop `.md`, `.txt` or `.pdf` files in `data/docs/` and `POST /ingest`.

## API

- `POST /ask` `{"question": "..."}` -> `{answer, abstained, sources[{id, source, page, score, snippet}], latency_ms}`
- `POST /ingest` re-index `data/docs`
- `GET /health`

## Evaluation

```bash
python -m eval.run_eval          # writes eval/results.md
```

`eval/dataset.jsonl` has 19 questions (16 answerable, 3 unanswerable) over the sample docs. Metrics:

- **Retrieval**: hit@4 and MRR, reported *before* and *after* reranking
- **Answer correctness**: reference keyword present in the answer
- **Answer relevance**: cosine similarity between question and answer embeddings
- **Faithfulness**: fraction of answer claims entailed by the retrieved context (NLI cross-encoder `nli-deberta-v3-small`); **hallucination rate = 1 - faithfulness**
- **Abstention accuracy** on unanswerable questions
- **Latency** p50 / p95

### Results

Measured on 19 questions (16 answerable, 3 unanswerable) over 4 synthetic policy documents, running on an 8 GB MacBook Air (Apple GPU via MPS) with `bge-small-en-v1.5`, `ms-marco-MiniLM-L-6-v2` and `Qwen2.5-1.5B-Instruct`.

| Metric | Result |
|---|---|
| hit@4 / MRR (hybrid only) | 1.00 / 1.00 |
| hit@4 / MRR (hybrid + rerank) | 1.00 / 1.00 |
| Answer correctness | 0.94 (15 of 16) |
| Answer relevance | 0.79 |
| Faithfulness | 0.87 |
| Hallucination rate (1 - faithfulness) | 0.13 |
| Abstention accuracy (unanswerable) | 3 of 3 |
| Latency p50 / p95 | 1.1 s / 1.7 s |

Notes on reading these numbers:

- Retrieval is saturated on a 4-document corpus, so this eval cannot show the benefit of reranking. A larger corpus is the next step.
- Faithfulness is a conservative estimate: each answer claim must be entailed by a single retrieved sentence (or adjacent pair) according to an NLI model, so claims that combine several sentences can be under-counted.
- Faithfulness is not correctness. An answer can be faithful to the retrieved text and still answer the wrong question.

### Known failure modes

- "What is the 401(k) match?" returns "$1,500" (the learning budget). Both facts are in the same chunk and the generator picks the wrong figure. Fix candidates: smaller chunks, a stronger generator, or answer verification.
- Correctness uses keyword matching, which is strict and small-sample. An LLM-judged check would be more robust.

## Tests

```bash
pytest -q
```

Tests run fully offline using lightweight fallbacks (`EMBEDDING_BACKEND=hash`, `RERANKER_BACKEND=lexical`, `GENERATOR_BACKEND=extractive`, `FAITHFULNESS_BACKEND=lexical`). These fallbacks exist for CI only; their retrieval and abstention quality is much weaker than the real models, so report numbers from the default configuration.

## Configuration

All settings are environment variables (see `app/config.py`): `EMBEDDING_MODEL`, `RERANKER_MODEL`, `GENERATOR_BACKEND` (`hf_local` | `hf_api` | `extractive`), `CHUNK_SIZE`, `RETRIEVE_K`, `RERANK_TOP_N`, etc.

## Limitations / next steps

- Abstention threshold is a fixed reranker score; tune it on a larger eval set.
- Eval set is small and synthetic; extend with real documents and questions.
- Local generation uses the Apple GPU (MPS) when available; on CPU-only machines expect slower responses.
