import re
from typing import Dict, List

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in and or is are was for on with what how who when does do can i my be by at it this that as from".split())


def _toks(t: str):
    return {w for w in _TOKEN.findall(t.lower()) if w not in _STOP and len(w) > 1}


class LexicalReranker:
    """Offline fallback: fraction of query terms covered by the passage."""

    abstain_threshold = 0.4

    def score(self, query: str, texts: List[str]) -> List[float]:
        q = _toks(query)
        return [len(q & _toks(t)) / max(len(q), 1) for t in texts]


class CrossEncoderReranker:
    # ms-marco logits: > 0 is generally "relevant"
    abstain_threshold = 0.0

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: List[str]) -> List[float]:
        return [float(x) for x in self.model.predict([(query, t) for t in texts], show_progress_bar=False)]


def get_reranker(backend: str, model: str):
    return LexicalReranker() if backend == "lexical" else CrossEncoderReranker(model)


def rerank(reranker, query: str, hits: List[Dict], top_n: int) -> List[Dict]:
    scores = reranker.score(query, [h["text"] for h in hits])
    for h, s in zip(hits, scores):
        h["rerank_score"] = s
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_n]
