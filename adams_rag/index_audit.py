"""Stream a consistent embedding/graph audit before considering a profile switch."""
import argparse
from collections import Counter
from contextlib import ExitStack
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3

from .evidence import read_snapshot
from .semantic_index import DEFAULT_EMBEDDING
from .storage import require_space


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit_profiles(db, db_path, directory, baseline, candidate):
    import numpy as np
    from usearch.index import Index
    if not baseline or not candidate or baseline == candidate:
        raise ValueError("Audit two different embedding profiles")
    directory = Path(directory)
    keys = {model: hashlib.sha256(model.encode()).hexdigest()[:16] for model in (baseline, candidate)}
    reports = {}
    with ExitStack() as stack:
        # Builders use these same locks for graph+manifest publication. Acquire
        # them before the database snapshot; do not block source ingestion.
        for key in sorted(keys.values()):
            path = directory / (key + ".lock")
            if path.exists():
                lock = stack.enter_context(path.open("r"))
                fcntl.flock(lock, fcntl.LOCK_SH)
        stack.enter_context(read_snapshot(db))
        identity = db.execute("SELECT value FROM collection_identity WHERE id=1").fetchone()[0]
        for model in (baseline, candidate):
            key = keys[model]
            errors, examples = Counter(), []
            def issue(reason, chunk_id=None):
                errors[reason] += 1
                if len(examples) < 20:
                    examples.append({"reason": reason, "chunk_id": chunk_id})
            generation = db.execute("SELECT generation FROM vector_generations WHERE model=?", (model,)).fetchone()
            generation = generation[0] if generation else 0
            fingerprint = hashlib.sha256(f"{Path(db_path).resolve()}:{identity}:{generation}".encode()).hexdigest()
            graph, manifest, graph_hash = None, {}, None
            try:
                manifest = json.loads((directory / (key + ".json")).read_text())
                if (not isinstance(manifest, dict) or manifest.get("model") != model
                        or manifest.get("generation") != fingerprint
                        or manifest.get("snapshot_generation") != generation):
                    issue("graph_manifest_not_current")
                graph = Index.restore(str(directory / (key + ".usearch")), view=True)
                if graph is None:
                    issue("graph_unavailable")
                else:
                    graph_hash = file_sha256(directory / (key + ".usearch"))
            except (OSError, ValueError, RuntimeError):
                issue("graph_or_manifest_unreadable")
            digest, scanned, dimension = hashlib.sha256(), 0, None
            cursor = db.execute("""SELECT v.*,c.text,d.title,d.status FROM dense_vectors v
                LEFT JOIN chunks c ON c.id=v.chunk_id LEFT JOIN documents d ON d.accession=c.accession
                WHERE v.model=? ORDER BY v.chunk_id""", (model,))
            while rows := cursor.fetchmany(256):
                vectors = graph.get([r["chunk_id"] for r in rows], dtype="f32") if graph is not None else [None] * len(rows)
                for row, stored in zip(rows, vectors):
                    scanned += 1
                    chunk_id = row["chunk_id"]
                    digest.update(str(chunk_id).encode() + b"\0" + row["content_hash"].encode() + b"\0" + row["vector"])
                    if row["text"] is None or row["status"] != "indexed":
                        issue("source_passage_unavailable", chunk_id)
                    elif hashlib.sha256((row["title"] + "\n" + row["text"]).encode()).hexdigest() != row["content_hash"]:
                        issue("embedding_input_hash_mismatch", chunk_id)
                    if dimension is None:
                        dimension = row["dimension"]
                    if (type(row["dimension"]) is not int or row["dimension"] < 1
                            or row["dimension"] != dimension or len(row["vector"]) != row["dimension"] * 4):
                        issue("invalid_vector_shape", chunk_id)
                        continue
                    vector = np.frombuffer(row["vector"], dtype=np.float32)
                    if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > .002:
                        issue("invalid_vector_values", chunk_id)
                    if stored is None:
                        issue("passage_missing_from_graph", chunk_id)
                    elif stored.shape != vector.shape or not np.allclose(stored, vector, rtol=0, atol=.0005):
                        issue("graph_vector_mismatch", chunk_id)
            if not scanned:
                issue("empty_profile")
            if graph is not None and len(graph) != scanned:
                issue("graph_vector_count_mismatch")
            if isinstance(manifest, dict) and manifest.get("dimension") != dimension:
                issue("graph_dimension_mismatch")
            reports[model] = {"vectors_scanned": scanned, "dimension": dimension,
                "vector_generation": generation, "snapshot_fingerprint": fingerprint,
                "graph_sha256": graph_hash,
                "stored_vectors_sha256": digest.hexdigest(), "errors": dict(errors), "examples": examples,
                "passed": not errors}
            del graph
        coverage = dict(db.execute("""SELECT count(*) AS baseline_vectors,
            coalesce(sum(v.chunk_id IS NULL),0) AS missing_passages,
            coalesce(sum(v.chunk_id IS NOT NULL AND v.content_hash<>b.content_hash),0) AS different_input_hashes
            FROM dense_vectors b LEFT JOIN dense_vectors v ON v.chunk_id=b.chunk_id AND v.model=?
            WHERE b.model=?""", (candidate, baseline)).fetchone())
    return {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_model": baseline, "candidate_model": candidate, "profiles": reports, "coverage": coverage,
        "coverage_and_integrity_passed": all(r["passed"] for r in reports.values()) and coverage["baseline_vectors"] > 0
            and coverage["missing_passages"] == 0 and coverage["different_input_hashes"] == 0,
        "graph_vector_absolute_tolerance": .0005, "retrieval_quality_verified": False,
        "source_fidelity_verified": False, "full_archive_verified": False}


def main(argv=None):
    from .mlx_embeddings import MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research.sqlite")
    parser.add_argument("--directory", default="data/index")
    parser.add_argument("--baseline", default=DEFAULT_EMBEDDING)
    parser.add_argument("--candidate", default=MODEL)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    db_path = Path(args.db).resolve()
    db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        result = audit_profiles(db, db_path, args.directory, args.baseline, args.candidate)
    finally:
        db.close()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        require_space(output.parent)
        output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
