from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline

from app.config import get_settings

s = get_settings()
SentenceTransformer(s.embedding_model)
CrossEncoder(s.reranker_model)
CrossEncoder(s.nli_model)
pipeline("text2text-generation", model=s.generator_model)
print("models cached")
