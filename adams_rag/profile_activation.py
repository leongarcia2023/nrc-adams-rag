"""Select the GPU profile only after matching integrity and retrieval evidence."""
import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .index_audit import file_sha256
from .mlx_embeddings import MODEL
from .profile_regression import retrieval_fingerprint, validate_cases
from .semantic_index import DEFAULT_EMBEDDING, SemanticIndex, active_model
from .storage import require_space
from .store import Store


def activate(store, directory, audit, regression, cases):
    validate_cases(cases)
    directory = Path(directory)
    models = (DEFAULT_EMBEDDING, MODEL)
    case_hash = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    if (audit.get("baseline_model") != models[0] or audit.get("candidate_model") != models[1]
            or regression.get("baseline_model") != models[0] or regression.get("candidate_model") != models[1]
            or audit.get("coverage_and_integrity_passed") is not True
            or regression.get("state") != "evaluated" or regression.get("known_source_regressions_passed") is not True
            or regression.get("queries_sha256") != case_hash
            or len(regression.get("cases", [])) != len(cases)
            or any(case.get("passed") is not True for case in regression.get("cases", []))):
        raise ValueError("Matching, successful coverage/integrity and retrieval checks are required")
    if store.db.in_transaction:
        raise ValueError("Profile activation requires its own transaction")
    require_space(directory)
    indices = [SemanticIndex(store, directory, model=model) for model in models]
    with ExitStack() as stack:
        lock = stack.enter_context((directory / "activation.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index in sorted(indices, key=lambda i: i.key):
            lock = stack.enter_context((directory / (index.key + ".lock")).open("r"))
            fcntl.flock(lock, fcntl.LOCK_SH)
        with store.db:
            store.db.execute("BEGIN IMMEDIATE")
            if active_model(directory) != models[0]:
                raise ValueError("The active profile changed; re-evaluate the current transition")
            if retrieval_fingerprint(store.db) != regression.get("retrieval_fingerprint"):
                raise ValueError("The retrieval collection or its reviews changed; rerun paired retrieval")
            coverage = indices[1].coverage_against(models[0])
            if not coverage["baseline_vectors"] or coverage["missing_passages"] or coverage["different_input_hashes"]:
                raise ValueError("Candidate coverage changed; retain the existing active profile")
            for index in indices:
                profile = audit.get("profiles", {}).get(index.model, {})
                fingerprint = index.fingerprint()
                manifest = json.loads(index.manifest.read_text())
                digest = file_sha256(index.path)
                if (profile.get("passed") is not True or profile.get("errors") or profile.get("dimension") != 384
                        or profile.get("snapshot_fingerprint") != fingerprint
                        or regression.get("snapshot_fingerprints", {}).get(index.model) != fingerprint
                        or profile.get("graph_sha256") != digest
                        or regression.get("graph_sha256", {}).get(index.model) != digest
                        or manifest.get("generation") != fingerprint or manifest.get("model") != index.model):
                    raise ValueError("Embedding or graph state changed after validation; rerun the checks")
            require_space(directory)
            value = {"version": 1, "model": models[1]}
            descriptor, temporary = tempfile.mkstemp(prefix="active-profile-", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    stream.write(json.dumps(value) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, directory / "ACTIVE_MODEL.json")
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return {"state": "activated", "previous_model": models[0], "model": models[1],
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "queries_sha256": case_hash,
        "retrieval_fingerprint": regression["retrieval_fingerprint"],
        "snapshot_fingerprints": regression["snapshot_fingerprints"], "graph_sha256": regression["graph_sha256"],
        "full_archive_verified": False, "general_retrieval_accuracy_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research.sqlite")
    parser.add_argument("--directory", default="data/index")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--regression", required=True)
    parser.add_argument("--cases", default="examples/profile-regression.json")
    args = parser.parse_args(argv)
    audit = json.loads(Path(args.audit).read_text())
    regression = json.loads(Path(args.regression).read_text())
    cases = json.loads(Path(args.cases).read_text())
    store = Store(args.db)
    try:
        result = activate(store, args.directory, audit, regression, cases)
    finally:
        store.db.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
