import re
from typing import Dict, List

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_CITE = re.compile(r"\[\d+\]")


def split_claims(answer: str) -> List[str]:
    answer = _CITE.sub("", answer)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(s.strip()) > 10]


class LexicalFaithfulness:
    """Offline fallback: a claim is supported if >=70% of its content words appear in the contexts."""

    def score(self, answer: str, contexts: List[str]) -> float:
        ctx = set(_TOKEN.findall(" ".join(contexts).lower()))
        claims = split_claims(answer)
        if not claims:
            return 1.0
        ok = 0
        for c in claims:
            toks = [t for t in _TOKEN.findall(c.lower()) if len(t) > 2]
            ok += (sum(t in ctx for t in toks) / max(len(toks), 1)) >= 0.7
        return ok / len(claims)


class NLIFaithfulness:
    """Fraction of answer claims entailed by the retrieved context (NLI cross-encoder)."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        # label order for cross-encoder/nli-* models: contradiction, entailment, neutral
        self.entail_idx = 1

    def score(self, answer: str, contexts: List[str]) -> float:
        claims = split_claims(answer)
        if not claims:
            return 1.0
        premise = " ".join(contexts)
        logits = self.model.predict([(premise, c) for c in claims], show_progress_bar=False)
        return float((np.argmax(logits, axis=1) == self.entail_idx).mean())


def get_faithfulness(backend: str, model: str):
    return LexicalFaithfulness() if backend == "lexical" else NLIFaithfulness(model)


def answer_relevance(embedder, question: str, answer: str) -> float:
    """Cosine similarity between question and answer embeddings (embeddings are normalized)."""
    return float(np.dot(embedder.embed_query(question), embedder.embed([_CITE.sub("", answer)])[0]))


def keyword_correct(answer: str, keywords: List[str]) -> bool:
    a = answer.lower()
    return any(k.lower() in a for k in keywords)


def hit_rank(hits: List[Dict], source: str) -> int | None:
    for i, h in enumerate(hits, 1):
        if h["source"] == source:
            return i
    return None
