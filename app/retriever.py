import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

from .embeddings import get_embedder
from .reranker import get_reranker, rerank
from .vectorstore import VectorStore
from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(t: str):
    return _TOKEN.findall(t.lower())


def rrf(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion over several ranked id lists."""
    fused: Dict[str, float] = {}
    for lst in rank_lists:
        for r, cid in enumerate(lst):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r + 1)
    return fused


class HybridRetriever:
    """Dense (Chroma) + BM25 candidates fused with RRF, then cross-encoder reranking."""

    def __init__(self, s: Settings, store: VectorStore, embedder=None, reranker=None):
        self.s = s
        self.store = store
        self.embedder = embedder or get_embedder(s)
        self.reranker = reranker or get_reranker(s.reranker_backend, s.reranker_model)
        self.refresh()

    def refresh(self):
        self.chunks = {c.metadata["chunk_id"]: c for c in self.store.all_chunks()}
        ids = list(self.chunks)
        self._ids = ids
        self.bm25 = BM25Okapi([_tok(self.chunks[i].page_content) for i in ids]) if ids else None

    def candidates(self, query: str, k: int) -> List[Dict]:
        """Stage 1 only (no reranking): hybrid candidate generation."""
        if not self.chunks:
            return []
        dense = self.store.query(self.embedder.embed_query(query), k)
        dense_ids = [h["chunk_id"] for h in dense]
        scores = self.bm25.get_scores(_tok(query))
        sparse_ids = [self._ids[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k] if scores[i] > 0]
        fused = rrf([dense_ids, sparse_ids])
        out = []
        for cid, sc in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]:
            c = self.chunks[cid]
            out.append({"chunk_id": cid, "text": c.page_content, "source": c.metadata["source"], "page": c.metadata["page"], "fusion_score": sc})
        return out

    def retrieve(self, query: str, top_n: int | None = None, use_rerank: bool = True) -> List[Dict]:
        top_n = top_n or self.s.rerank_top_n
        cands = self.candidates(query, self.s.retrieve_k)
        if not cands:
            return []
        return rerank(self.reranker, query, cands, top_n) if use_rerank else cands[:top_n]
