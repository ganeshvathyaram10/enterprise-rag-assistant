import os
import tempfile

os.environ.update(
    EMBEDDING_BACKEND="hash",
    RERANKER_BACKEND="lexical",
    GENERATOR_BACKEND="extractive",
    FAITHFULNESS_BACKEND="lexical",
    CHROMA_DIR=tempfile.mkdtemp(),
)

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402
from app.ingest import chunk_documents, load_documents  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.retriever import rrf  # noqa: E402
from eval.metrics import LexicalFaithfulness  # noqa: E402


def test_chunking_adds_ids_and_respects_size():
    s = get_settings()
    chunks = chunk_documents(load_documents("data/docs"), s)
    assert len(chunks) > 8
    assert all("chunk_id" in c.metadata for c in chunks)
    assert max(len(c.page_content) for c in chunks) <= s.chunk_size + 50


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert fused["a"] > fused["c"] and fused["b"] > fused["d"]


def test_faithfulness_detects_unsupported_claim():
    f = LexicalFaithfulness()
    ctx = ["Remote employees receive a one-time stipend of $750."]
    assert f.score("Remote employees receive a stipend of $750.", ctx) == 1.0
    assert f.score("Employees also receive a company yacht and helicopter rides.", ctx) < 0.5


def test_api_answers_and_abstains():
    with TestClient(app) as c:
        assert c.get("/health").json()["chunks_indexed"] > 0
        r = c.post("/ask", json={"question": "What is the home office stipend for remote workers?"}).json()
        assert "750" in r["answer"] and r["sources"][0]["source"] == "remote_work_policy.md"
        r = c.post("/ask", json={"question": "Does Acme offer stock options to new hires?"}).json()
        assert r["abstained"] is True and r["sources"] == []
        assert c.post("/ask", json={"question": "hi"}).status_code == 422
