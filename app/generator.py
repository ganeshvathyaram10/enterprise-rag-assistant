import re
from typing import Dict, List

from .config import Settings

ABSTAIN = "I don't know based on the provided documents."

PROMPT = (
    "Answer the question using ONLY the context below. Cite sources as [1], [2] matching the context numbers. "
    "If the context does not contain the answer, reply exactly: " + ABSTAIN + "\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def build_context(hits: List[Dict]) -> str:
    return "\n\n".join(f"[{i}] ({h['source']}) {h['text']}" for i, h in enumerate(hits, 1))


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]


class ExtractiveGenerator:
    """No-LLM fallback: returns the best-matching sentences from the top chunk, with citations."""

    def generate(self, question: str, hits: List[Dict]) -> str:
        q = set(re.findall(r"[a-z0-9]+", question.lower()))
        scored = []
        for i, h in enumerate(hits[:2], 1):
            for s in _sentences(h["text"]):
                overlap = len(q & set(re.findall(r"[a-z0-9]+", s.lower())))
                scored.append((overlap, i, s))
        scored.sort(key=lambda x: -x[0])
        top = [x for x in scored[:2] if x[0] > 0]
        return " ".join(f"{s} [{i}]" for _, i, s in top) or ABSTAIN


class HFLocalGenerator:
    def __init__(self, model_name: str):
        from transformers import pipeline

        self.pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, question: str, hits: List[Dict]) -> str:
        prompt = PROMPT.format(context=build_context(hits), question=question)
        out = self.pipe(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"].strip()
        return out or ABSTAIN


class HFApiGenerator:
    """Hugging Face Inference API (needs HF_TOKEN)."""

    def __init__(self, model_name: str):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(model=model_name)

    def generate(self, question: str, hits: List[Dict]) -> str:
        prompt = PROMPT.format(context=build_context(hits), question=question)
        r = self.client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=0.0)
        return r.choices[0].message.content.strip()


def get_generator(s: Settings):
    if s.generator_backend == "extractive":
        return ExtractiveGenerator()
    if s.generator_backend == "hf_api":
        return HFApiGenerator(s.hf_api_model)
    return HFLocalGenerator(s.generator_model)
