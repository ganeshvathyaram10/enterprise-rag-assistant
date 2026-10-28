from typing import Dict, List

import chromadb
import numpy as np
from langchain_core.documents import Document


class VectorStore:
    """Thin wrapper over a persistent Chroma collection (cosine space)."""

    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def reset(self):
        self.client.delete_collection(self.collection_name)
        self.col = self.client.get_or_create_collection(self.collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: List[Document], embeddings: np.ndarray):
        ids = [c.metadata["chunk_id"] for c in chunks]
        metas = [{"source": c.metadata["source"], "page": int(c.metadata["page"])} for c in chunks]
        for i in range(0, len(ids), 500):
            self.col.add(
                ids=ids[i : i + 500],
                documents=[c.page_content for c in chunks[i : i + 500]],
                embeddings=embeddings[i : i + 500].tolist(),
                metadatas=metas[i : i + 500],
            )

    def count(self) -> int:
        return self.col.count()

    def all_chunks(self) -> List[Document]:
        res = self.col.get(include=["documents", "metadatas"])
        out = []
        for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            out.append(Document(page_content=text, metadata={**meta, "chunk_id": cid}))
        return out

    def query(self, embedding: np.ndarray, k: int) -> List[Dict]:
        k = min(k, max(self.count(), 1))
        res = self.col.query(query_embeddings=[embedding.tolist()], n_results=k, include=["documents", "metadatas", "distances"])
        hits = []
        for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"chunk_id": cid, "text": text, "source": meta["source"], "page": meta["page"], "score": 1.0 - dist})
        return hits
