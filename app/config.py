import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    docs_dir: str = os.getenv("DOCS_DIR", "data/docs")
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    collection: str = os.getenv("COLLECTION", "enterprise_docs")
    # "st" = sentence-transformers, "hash" = offline deterministic fallback (tests/CI)
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "st")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    # "cross_encoder" or "lexical" (offline fallback)
    reranker_backend: str = os.getenv("RERANKER_BACKEND", "cross_encoder")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # "hf_local" | "hf_api" | "extractive"
    generator_backend: str = os.getenv("GENERATOR_BACKEND", "hf_local")
    generator_model: str = os.getenv("GENERATOR_MODEL", "google/flan-t5-base")
    hf_api_model: str = os.getenv("HF_API_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    # "nli" | "lexical"
    faithfulness_backend: str = os.getenv("FAITHFULNESS_BACKEND", "nli")
    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    retrieve_k: int = int(os.getenv("RETRIEVE_K", "20"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "4"))


def get_settings() -> Settings:
    return Settings()
