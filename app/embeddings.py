import hashlib
import re
from typing import List

import numpy as np

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. Offline fallback for tests/CI only."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        # BGE models work best with this query instruction
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0]


def get_embedder(s: Settings):
    if s.embedding_backend == "hash":
        return HashEmbedder()
    return STEmbedder(s.embedding_model)
