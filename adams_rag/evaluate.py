"""Retrieval evaluation with external relevance labels. Never trains on test labels."""
import json
import math
import re
import time
from pathlib import Path

from .research import rank_candidates


def evaluate(store, queries_path, labels_path, semantic=None, k=10):
    queries = [json.loads(line) for line in Path(queries_path).read_text().splitlines() if line.strip()]
    labels = {}
    for line in Path(labels_path).read_text().splitlines():
        if not line.strip() or line.startswith("query-id"):
            continue
        qid, docid, score = line.split("\t")
        if float(score) > 0:
            match = re.search(r"ML\d{5}[A-Za-z0-9]\d{3}", docid, re.I)
            labels.setdefault(qid, set()).add(match.group(0).upper() if match else docid)
    results = []
    for q in queries:
        relevant = labels.get(q["_id"])
        if not relevant:
            continue
        start = time.monotonic()
        sources = rank_candidates(store, q["text"], semantic=semantic)
        ranked = list(dict.fromkeys(s["accession"] for s in sources))[:k]
        hits = [1 if r in relevant else 0 for r in ranked]
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        results.append({"query_id": q["_id"], "question": q["text"], "ranked": ranked,
                        "recall": sum(hits) / len(relevant), "ndcg": dcg / ideal,
                        "reciprocal_rank": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0),
                        "seconds": time.monotonic() - start})
    if not results:
        raise ValueError("No labelled queries to evaluate")
    return {"queries": len(results), "k": k, "mode": "hybrid" if semantic else "keyword",
            "recall_at_k": sum(r["recall"] for r in results) / len(results),
            "ndcg_at_k": sum(r["ndcg"] for r in results) / len(results),
            "mrr_at_k": sum(r["reciprocal_rank"] for r in results) / len(results),
            "mean_seconds": sum(r["seconds"] for r in results) / len(results),
            "generation_evaluated": False, "details": results}
