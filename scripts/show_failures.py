import json
from app.pipeline import RAGPipeline

p = RAGPipeline()
p.ingest()
for line in open("eval/dataset.jsonl"):
    r = json.loads(line)
    a = p.ask(r["question"])
    if r["answerable"]:
        ok = any(k.lower() in a["answer"].lower() for k in r["keywords"])
    else:
        ok = a["abstained"]
    if not ok:
        print(f"\nQ: {r['question']}\nA: {a['answer']}\nexpected: {r['keywords']}")
