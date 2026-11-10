"""Paired known-source retrieval checks over the entire current local collection."""
import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import time

from .evidence import read_snapshot
from .index_audit import file_sha256
from .research import rank_candidates, validate_filters
from .semantic_index import DEFAULT_EMBEDDING, SemanticIndex
from .storage import require_space
from .store import Store


def retrieval_fingerprint(db):
    identity = db.execute("SELECT value FROM collection_identity WHERE id=1").fetchone()[0]
    generation = db.execute("SELECT generation FROM retrieval_state WHERE id=1").fetchone()[0]
    return hashlib.sha256(f"{identity}:{generation}".encode()).hexdigest()


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128:
        raise ValueError("Use 1–128 labelled regression queries")
    ids = set()
    for case in cases:
        if (not isinstance(case, dict) or set(case) - {"id", "question", "relevant", "docket", "date_from", "date_to"}
                or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids
                or not isinstance(case.get("question"), str) or not 1 <= len(case["question"].strip()) <= 2000
                or not isinstance(case.get("relevant"), list) or not case["relevant"]
                or any(not isinstance(value, str) or not value for value in case["relevant"])):
            raise ValueError("Each regression query needs a unique ID, question and relevant source IDs")
        validate_filters(case.get("docket"), case.get("date_from"), case.get("date_to"))
        ids.add(case["id"])


def compare_retrieval(store, baseline, candidate, cases, k=10):
    validate_cases(cases)
    if type(k) is not int or not 1 <= k <= 100 or baseline.model == candidate.model:
        raise ValueError("Use two different profiles and 1–100 ranked documents")
    if baseline.store is not store or candidate.store is not store:
        raise ValueError("Paired retrieval must use one collection snapshot")
    result = {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "baseline_model": baseline.model,
        "candidate_model": candidate.model, "queries_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "k": k, "cases": [], "known_source_regressions_passed": False,
        "independent_quality_benchmark": False, "generation_evaluated": False, "full_archive_verified": False}
    with ExitStack() as stack:
        for index in sorted((baseline, candidate), key=lambda i: i.key):
            path = index.directory / (index.key + ".lock")
            if path.exists():
                lock = stack.enter_context(path.open("r"))
                fcntl.flock(lock, fcntl.LOCK_SH)
        stack.enter_context(read_snapshot(store.db))
        result["coverage"] = candidate.coverage_against(baseline.model)
        coverage = result["coverage"]
        if not coverage["baseline_vectors"] or coverage["missing_passages"] or coverage["different_input_hashes"]:
            return {**result, "state": "coverage_pending"}
        result["snapshot_fingerprints"] = {index.model: index.fingerprint() for index in (baseline, candidate)}
        result["retrieval_fingerprint"] = retrieval_fingerprint(store.db)
        result["graph_sha256"] = {}
        for index in (baseline, candidate):
            try:
                manifest = json.loads(index.manifest.read_text())
                if manifest.get("generation") != result["snapshot_fingerprints"][index.model] or manifest.get("model") != index.model:
                    return {**result, "state": "graph_publication_pending"}
                result["graph_sha256"][index.model] = file_sha256(index.path)
            except (OSError, ValueError, AttributeError):
                return {**result, "state": "graph_publication_pending"}
        result["collection_passages"] = store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        for case in cases:
            pair = {"id": case["id"], "question": case["question"], "relevant": case["relevant"]}
            passage_ids = {}
            for label, index in (("baseline", baseline), ("candidate", candidate)):
                started = time.monotonic()
                exclusions = []
                ranked = rank_candidates(store, case["question"], semantic=index, docket=case.get("docket"),
                    date_from=case.get("date_from"), date_to=case.get("date_to"), exclusions=exclusions)
                documents = list(dict.fromkeys(row["accession"] for row in ranked))[:k]
                pair[label] = {"documents": documents, "relevant_ranks": {
                    source: documents.index(source) + 1 if source in documents else None for source in case["relevant"]},
                    "seconds": round(time.monotonic() - started, 4), "excluded_evidence": exclusions}
                passage_ids[label] = {row["id"] for row in ranked[:10]}
            old, new = pair["baseline"]["relevant_ranks"], pair["candidate"]["relevant_ranks"]
            pair["passed"] = all(new[source] is not None and (old[source] is None or new[source] <= old[source]) for source in new)
            pair["top10_passage_overlap"] = len(passage_ids["baseline"] & passage_ids["candidate"]) / max(1, len(passage_ids["baseline"]))
            result["cases"].append(pair)
    return {**result, "state": "evaluated", "known_source_regressions_passed": all(c["passed"] for c in result["cases"]),
        "limitations": ["Source-derived labels check known behavior; they are not independent relevance judgments.",
                        "A profile switch also requires a current vector/graph integrity audit.",
                        "This does not establish general retrieval accuracy, source fidelity or answer correctness."]}


class QueryCache:
    def __init__(self, questions, vectors):
        import numpy as np
        vectors = list(vectors)
        if len(vectors) != len(questions):
            raise ValueError("Query embedding count mismatch")
        self.vectors = {}
        for question, vector in zip(questions, vectors):
            vector = np.asarray(vector, dtype=np.float32)
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError("Invalid query embedding")
            self.vectors[question] = vector

    def query_embed(self, question):
        return iter([self.vectors[question]])


def main(argv=None):
    from .mlx_embeddings import MODEL, GuardedEncoder
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research.sqlite")
    parser.add_argument("--directory", default="data/index")
    parser.add_argument("--cases", default="examples/profile-regression.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    cases = json.loads(Path(args.cases).read_text())
    validate_cases(cases)
    store = Store(args.db)
    try:
        baseline = SemanticIndex(store, args.directory, model=DEFAULT_EMBEDDING)
        candidate = SemanticIndex(store, args.directory, model=MODEL)
        coverage = candidate.coverage_against(baseline.model)
        if coverage["baseline_vectors"] and not coverage["missing_passages"] and not coverage["different_input_hashes"]:
            questions = list(dict.fromkeys(case["question"] for case in cases))
            # Both current GPU query/passage methods use the same tokenization.
            # Encode the query batch before opening the collection read snapshot.
            candidate.encoder = QueryCache(questions, GuardedEncoder(args.directory).passage_embed(questions))
            baseline.encoder = QueryCache(questions, baseline._encoder().query_embed(questions))
        result = compare_retrieval(store, baseline, candidate, cases)
    finally:
        store.db.close()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        require_space(output.parent)
        output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
