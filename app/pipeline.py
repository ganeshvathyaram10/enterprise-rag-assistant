import time
from typing import Dict, Optional

from .config import Settings, get_settings
from .embeddings import get_embedder
from .generator import ABSTAIN, get_generator
from .ingest import chunk_documents, load_documents
from .reranker import get_reranker
from .retriever import HybridRetriever
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, s: Optional[Settings] = None):
        self.s = s or get_settings()
        self.embedder = get_embedder(self.s)
        self.store = VectorStore(self.s.chroma_dir, self.s.collection)
        self.reranker = get_reranker(self.s.reranker_backend, self.s.reranker_model)
        self.retriever = HybridRetriever(self.s, self.store, self.embedder, self.reranker)
        self.generator = get_generator(self.s)

    def ingest(self, docs_dir: Optional[str] = None, reset: bool = True) -> Dict:
        if reset:
            self.store.reset()
        docs = load_documents(docs_dir or self.s.docs_dir)
        chunks = chunk_documents(docs, self.s)
        if chunks:
            self.store.add(chunks, self.embedder.embed([c.page_content for c in chunks]))
        self.retriever.refresh()
        return {"documents": len(docs), "chunks": len(chunks)}

    def ask(self, question: str) -> Dict:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question)
        t_ret = time.perf_counter()
        # Abstain when nothing relevant was retrieved (cheaper and safer than letting the LLM guess).
        if not hits or hits[0]["rerank_score"] < self.reranker.abstain_threshold:
            return {"answer": ABSTAIN, "abstained": True, "sources": [], "latency_ms": {"retrieval": round((t_ret - t0) * 1000, 1), "generation": 0.0}}
        answer = self.generator.generate(question, hits)
        t_gen = time.perf_counter()
        return {
            "answer": answer,
            "abstained": answer.strip().startswith("I don't know"),
            "sources": [{"id": i, "source": h["source"], "page": h["page"], "score": round(h["rerank_score"], 3), "snippet": h["text"][:200]} for i, h in enumerate(hits, 1)],
            "contexts": [h["text"] for h in hits],
            "latency_ms": {"retrieval": round((t_ret - t0) * 1000, 1), "generation": round((t_gen - t_ret) * 1000, 1)},
        }
