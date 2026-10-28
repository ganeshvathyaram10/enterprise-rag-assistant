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
    """Local instruction-tuned chat model (e.g. Qwen2.5-1.5B-Instruct)."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()

    def generate(self, question: str, hits: List[Dict]) -> str:
        nl = chr(10)
        system = (
            "You answer questions using only the provided context. "
            "Answer in one complete sentence that states the specific facts "
            "(numbers, limits, names), for example: Hybrid employees must be in the office 3 days per week [1]. "
            "Cite the context numbers like [1]. "
            "If the context does not contain the answer, reply exactly: " + ABSTAIN
        )
        user = "Context:" + nl + build_context(hits) + nl + nl + "Question: " + question
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        answer = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return answer or ABSTAIN


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
