import re
from typing import Dict, List

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_CITE = re.compile(r"\[\d+\]")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_claims(answer: str) -> List[str]:
    answer = _CITE.sub("", answer)
    return [s.strip() for s in _SENT.split(answer) if len(s.strip()) > 3]


def premise_windows(contexts: List[str]) -> List[str]:
    """Single sentences plus adjacent sentence pairs from the retrieved chunks."""
    out: List[str] = []
    for c in contexts:
        c = re.sub(r"^#+\s*", "", c, flags=re.M)
        sents = [s.strip() for s in _SENT.split(c) if len(s.strip()) > 10]
        out.extend(sents)
        out.extend(a + " " + b for a, b in zip(sents, sents[1:]))
    return out


class LexicalFaithfulness:
    """Offline fallback: a claim is supported if >=70% of its content words appear in the contexts."""

    def score(self, answer: str, contexts: List[str]) -> float:
        ctx = set(_TOKEN.findall(" ".join(contexts).lower()))
        claims = split_claims(answer)
        if not claims:
            return 0.0
        ok = 0
        for c in claims:
            toks = [t for t in _TOKEN.findall(c.lower()) if len(t) > 2]
            ok += (sum(t in ctx for t in toks) / max(len(toks), 1)) >= 0.7
        return ok / len(claims)


class NLIFaithfulness:
    """Fraction of answer claims entailed by at least one retrieved sentence (or sentence pair)."""

    def __init__(self, model_name: str, threshold: float = 0.5):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        labels = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        self.entail_idx = next(i for i, v in labels.items() if "entail" in v)
        self.threshold = threshold

    def score(self, answer: str, contexts: List[str]) -> float:
        claims = split_claims(answer)
        premises = premise_windows(contexts)
        if not claims or not premises:
            return 0.0
        supported = 0
        for claim in claims:
            logits = np.asarray(self.model.predict([(p, claim) for p in premises], show_progress_bar=False))
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            supported += probs[:, self.entail_idx].max() >= self.threshold
        return supported / len(claims)


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
