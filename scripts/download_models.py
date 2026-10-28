from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import get_settings

s = get_settings()
SentenceTransformer(s.embedding_model)
CrossEncoder(s.reranker_model)
CrossEncoder(s.nli_model)
AutoTokenizer.from_pretrained(s.generator_model)
AutoModelForCausalLM.from_pretrained(s.generator_model)
print("models cached")
