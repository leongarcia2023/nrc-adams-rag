"""Continuously acquire catalog text with durable coverage and storage stops."""
import argparse
from contextlib import nullcontext
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from .backfill import backfill
from .client import AdamsClient, TemporaryRequestError
from .config import load_api_key
from .document_prefetch import DocumentPrefetchClient
from .storage import StoragePause, require_space, reserve_gb
from .store import Store
from .transport import PersistentOpener


def resume_at(rate, scope, daily_limit, now=None):
    now = time.time() if now is None else now
    count, oldest = rate.db.execute(
        "SELECT count(*),min(at) FROM http_requests WHERE scope=? AND at>?", (scope, now - 86400)).fetchone()
    next_at = rate.db.execute("SELECT next_at FROM http_state WHERE scope=?", (scope,)).fetchone()[0]
    return max(now + 5, next_at, oldest + 86401 if count >= daily_limit and oldest is not None else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research.sqlite")
    parser.add_argument("--catalog", default="../adams-catalog/catalog.sqlite")
    parser.add_argument("--rate-db", default="data/adams.sqlite")
    parser.add_argument("--directory", default="data/archive")
    parser.add_argument("--library", choices=["main", "legacy", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--rps", type=float, default=2)
    parser.add_argument("--daily-limit", type=int, default=90000)
    parser.add_argument("--min-free-gb", type=float)
    parser.add_argument("--prefetch", action="store_true", help="Overlap main-library downloads with local ingestion")
    parser.add_argument("--prefetch-depth", type=int, choices=(1, 2), default=1,
                        help="Bounded lookahead and connection count; request-rate ceilings remain shared")
    args = parser.parse_args(argv)
    if args.prefetch_depth != 1 and not args.prefetch:
        parser.error("Use --prefetch with --prefetch-depth")
    try:
        args.min_free_gb = reserve_gb(args.min_free_gb)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.batch_size <= 10000 or not 0 < args.rps <= 6 or args.daily_limit < 1:
        parser.error("Use 1–10,000 documents/batch, 0–6 requests/sec and a positive daily budget")
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An archive worker is already running")
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    def cancelled():
        return stopping[0] or (directory / "STOP").exists()
    store, rate = Store(args.db), Store(args.rate_db)
    transport = PersistentOpener()
    key = load_api_key()
    started, last_report = time.monotonic(), [0]
    initial = store.db.execute("SELECT count(*) FROM documents WHERE status='indexed'").fetchone()[0]
    current = {"library": None, "accession": None, "document_lookahead": args.prefetch,
               "document_lookahead_depth": args.prefetch_depth if args.prefetch else 0,
               "last_batch_prefetch": None}
    def report(state, detail=None, retry_at=None):
        # This report needs only progress totals. Full status also checks source
        # quality and other diagnostics, which can stall acquisition as the
        # collection grows even though none of those results are used here.
        indexed = store.db.execute("SELECT count(*) FROM documents WHERE status='indexed'").fetchone()[0]
        chunks = store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        vectors = store.db.execute("SELECT count(*) FROM dense_vectors").fetchone()[0]
        exists = store.db.execute("SELECT 1 FROM sqlite_master WHERE name='backfill_totals'").fetchone()
        totals = [dict(r) for r in store.db.execute(
            "SELECT c.library,t.outcome,sum(t.count) AS count FROM backfill_totals t JOIN backfill_cursors c USING(job) GROUP BY c.library,t.outcome")] if exists else []
        cursors = [dict(r) for r in store.db.execute(
            "SELECT library,last_row,state,updated FROM backfill_cursors")] if exists else []
        import shutil
        value = {"state": state, "pid": os.getpid(), **current,
                 "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "elapsed_seconds": round(time.monotonic() - started), "detail": detail,
                 "retry_at_utc": dt.datetime.fromtimestamp(retry_at, dt.timezone.utc).isoformat() if retry_at else None,
                 "indexed_documents": indexed,
                 "new_documents_this_run": indexed - initial,
                 "chunks": chunks, "vectors": vectors,
                 "free_bytes": shutil.disk_usage(store.path.parent).free,
                 "reserve_bytes": int(args.min_free_gb * 1e9), "coverage": totals, "cursors": cursors,
                 "full_archive_verified": False, "local_rps_ceiling": args.rps, "local_daily_budget": args.daily_limit}
        temp = directory / "STATUS.json.tmp"
        temp.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temp, directory / "STATUS.json")
        print(json.dumps(value), flush=True)
        last_report[0] = time.monotonic()
    def progress(accession, done, total):
        current["accession"] = accession
        if time.monotonic() - last_report[0] >= 30:
            report("running")
    def wait_until(deadline):
        # Small waits remain responsive to signals, stop files, and storage pressure.
        while time.time() < deadline and not cancelled():
            require_space(store.path.parent, args.min_free_gb)
            time.sleep(min(5, max(0, deadline - time.time())))
    try:
        require_space(store.path.parent, args.min_free_gb)
        report("starting")
        libraries = ["main", "legacy"] if args.library == "both" else [args.library]
        failures = 0
        # Alternate bounded batches so legacy text acquisition does not wait
        # behind millions of main-library requests. Both use the same rate gate.
        while libraries and not cancelled():
            library = libraries[0]
            current["library"] = library
            require_space(store.path.parent, args.min_free_gb)
            lookahead = args.prefetch and library == "main"
            if lookahead:
                client = DocumentPrefetchClient(rate, key, max_requests=args.batch_size * 6 + 10,
                    rps=args.rps, daily_limit=args.daily_limit, storage_path=store.path.parent,
                    min_free_gb=args.min_free_gb, cancelled=cancelled, lookahead_depth=args.prefetch_depth)
            else:
                client = AdamsClient(rate, key, max_requests=args.batch_size * 6 + 10,
                    rps=args.rps, daily_limit=args.daily_limit, opener=transport,
                    include_search_content=False, cancelled=cancelled)
            try:
                with client if lookahead else nullcontext(client):
                    result = backfill(store, client, args.catalog, args.batch_size, library,
                                      args.min_free_gb, progress=progress, cancelled=cancelled)
                current["last_batch_prefetch"] = dict(client.stats) if lookahead else None
            except TemporaryRequestError:
                failures += 1
                deadline = time.time() + min(3600, 300 * 2 ** min(failures - 1, 4))
                report("network_wait", "Temporary NRC/network failures; cursor retained", deadline)
                wait_until(deadline)
                continue
            failures = 0
            if result["state"] in {"storage_pause", "stopped"}:
                report(result["state"], "; ".join(result["warnings"]) or None)
                return
            if result["state"] == "catalog_exhausted":
                report("catalog_exhausted", "Catalog scanned; missing text and unsupported records remain coverage gaps")
                libraries.pop(0)
            elif result["state"] == "budget_pause":
                deadline = resume_at(rate, client.scope, args.daily_limit)
                report("budget_wait", "; ".join(result["warnings"]), deadline)
                wait_until(deadline)
            else:
                report("running")
                libraries.append(libraries.pop(0))
        if cancelled():
            report("stopped", "Committed progress retained")
            return
        report("scan_finished", "Acquisition scan finished; full coverage and answer validation are not certified")
    except StoragePause as error:
        report("storage_pause", str(error))
    except Exception as error:
        # Do not log arbitrary exception bodies or tracebacks containing credentials.
        report("needs_review", type(error).__name__)
        raise SystemExit(1) from None
    finally:
        transport.close()
        store.db.close()
        rate.db.close()
        lock.close()


if __name__ == "__main__":
    main()
