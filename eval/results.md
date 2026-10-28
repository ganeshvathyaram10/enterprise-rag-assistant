# Evaluation results

```json
{
  "n_questions": 19,
  "hit@4_hybrid_only": 1.0,
  "hit@4_hybrid+rerank": 1.0,
  "mrr_hybrid_only": 1.0,
  "mrr_hybrid+rerank": 1.0,
  "answer_correctness": 0.938,
  "answer_relevance": 0.787,
  "faithfulness": 0.867,
  "hallucination_rate": 0.133,
  "abstention_accuracy_on_unanswerable": 1.0,
  "latency_p50_ms": 1126.1,
  "latency_p95_ms": 1700.3,
  "config": {
    "embedding": "BAAI/bge-small-en-v1.5",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generator": "Qwen/Qwen2.5-1.5B-Instruct",
    "faithfulness": "cross-encoder/nli-deberta-v3-small"
  }
}
```
