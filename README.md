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
| Generator | `google/flan-t5-base` (local) or HF Inference API | swap via `GENERATOR_BACKEND` |
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

Run `python -m eval.run_eval` with the default (real) models and paste the output here:

| Metric | Hybrid only | Hybrid + rerank |
|---|---|---|
| hit@4 | _fill in_ | _fill in_ |
| MRR | _fill in_ | _fill in_ |
| Faithfulness | | _fill in_ |
| Hallucination rate | | _fill in_ |
| Abstention accuracy | | _fill in_ |
| p95 latency (ms) | | _fill in_ |

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
- flan-t5-base is weak at long synthesis; use `hf_api` or a bigger model for production quality.
