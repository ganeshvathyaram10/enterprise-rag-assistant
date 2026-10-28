"""Evaluate retrieval quality, answer relevance and faithfulness/hallucination.

Usage:  python -m eval.run_eval [--out eval/results.md]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.pipeline import RAGPipeline
from .metrics import answer_relevance, get_faithfulness, hit_rank, keyword_correct

K = 4


def load(path="eval/dataset.jsonl"):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def retrieval_stats(pipe, rows, use_rerank):
    ranks = []
    for r in rows:
        hits = pipe.retriever.retrieve(r["question"], top_n=K, use_rerank=use_rerank)
        ranks.append(hit_rank(hits, r["source"]))
    hit = np.mean([x is not None for x in ranks])
    mrr = np.mean([1 / x if x else 0 for x in ranks])
    return float(hit), float(mrr)


def main(out):
    pipe = RAGPipeline()
    pipe.ingest()
    rows = load()
    answerable = [r for r in rows if r["answerable"]]
    faith = get_faithfulness(pipe.s.faithfulness_backend, pipe.s.nli_model)

    h0, m0 = retrieval_stats(pipe, answerable, use_rerank=False)
    h1, m1 = retrieval_stats(pipe, answerable, use_rerank=True)

    correct, rel, fa, lat, abst_ok = [], [], [], [], []
    for r in rows:
        t = time.perf_counter()
        res = pipe.ask(r["question"])
        lat.append((time.perf_counter() - t) * 1000)
        if not r["answerable"]:
            abst_ok.append(res["abstained"])
            continue
        if res["abstained"]:
            correct.append(False)  # wrongly refused an answerable question
            continue
        correct.append(keyword_correct(res["answer"], r["keywords"]))
        rel.append(answer_relevance(pipe.embedder, r["question"], res["answer"]))
        fa.append(faith.score(res["answer"], res["contexts"]))

    report = {
        "n_questions": len(rows),
        f"hit@{K}_hybrid_only": round(h0, 3),
        f"hit@{K}_hybrid+rerank": round(h1, 3),
        "mrr_hybrid_only": round(m0, 3),
        "mrr_hybrid+rerank": round(m1, 3),
        "answer_correctness": round(float(np.mean(correct)), 3),
        "answer_relevance": round(float(np.mean(rel)), 3) if rel else None,
        "faithfulness": round(float(np.mean(fa)), 3) if fa else None,
        "hallucination_rate": round(1 - float(np.mean(fa)), 3) if fa else None,
        "abstention_accuracy_on_unanswerable": round(float(np.mean(abst_ok)), 3) if abst_ok else None,
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
        "config": {
            "embedding": pipe.s.embedding_model if pipe.s.embedding_backend == "st" else "hash(offline)",
            "reranker": pipe.s.reranker_model if pipe.s.reranker_backend == "cross_encoder" else "lexical(offline)",
            "generator": pipe.s.generator_model if pipe.s.generator_backend == "hf_local" else pipe.s.generator_backend,
            "faithfulness": pipe.s.nli_model if pipe.s.faithfulness_backend == "nli" else "lexical(offline)",
        },
    }
    print(json.dumps(report, indent=2))
    Path(out).write_text("# Evaluation results\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results.md")
    main(ap.parse_args().out)
