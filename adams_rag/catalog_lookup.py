"""Read catalog metadata by accession or NRC record ID without merging legacy records."""
import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import zlib

from .storage import StoragePause, require_space

INDEX = "item_accession_lookup_v1"


def connect(path, writable=False):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("The catalog file is unavailable")
    db = sqlite3.connect(path.as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


def index_ready(db):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (INDEX,)).fetchone())


def build_index(path, min_free_gb=None, cancelled=None, progress=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("The catalog file is unavailable")
    with path.with_suffix(".lookup.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A catalog lookup index build is already running") from None
        db = connect(path, writable=True)
        failure = [None]
        try:
            if index_ready(db):
                return {"state": "ready", "created": False}
            rows = db.execute("SELECT final_count,finished FROM libraries").fetchall()
            if not rows or any(r[0] is None or not r[1] for r in rows):
                raise ValueError("Build lookup indexes after catalog reconciliation")
            estimate = max(64 * 1024 * 1024, sum(r[0] for r in rows) * 144)
            require_space(path.parent, min_free_gb, extra_bytes=estimate)
            require_space(tempfile.gettempdir(), min_free_gb, extra_bytes=estimate)
            if cancelled and cancelled():
                raise InterruptedError("Catalog lookup indexing cancelled")
            last_check = [0]
            started = time.monotonic()
            def guard():
                now = time.monotonic()
                if now - last_check[0] < .5:
                    return 0
                last_check[0] = now
                try:
                    if cancelled and cancelled():
                        raise InterruptedError("Catalog lookup indexing cancelled")
                    require_space(path.parent, min_free_gb)
                    require_space(tempfile.gettempdir(), min_free_gb)
                    if progress:
                        progress(round(now - started, 1))
                    return 0
                except (StoragePause, InterruptedError) as error:
                    failure[0] = error
                    return 1
            db.set_progress_handler(guard, 10000)
            try:
                # SQLite builds this atomically. Readers retain the existing catalog;
                # cancellation rolls back the new index without changing its records.
                with db:
                    db.execute(f"CREATE INDEX {INDEX} ON items(accession COLLATE NOCASE,library)")
            except sqlite3.OperationalError:
                if failure[0]:
                    raise failure[0] from None
                raise
            finally:
                db.set_progress_handler(None, 0)
            return {"state": "ready", "created": True, "seconds": round(time.monotonic() - started, 2)}
        finally:
            db.close()


def normalize_query(accession=None, library="both", record_id=None, limit=50, cursor=None):
    if library not in {"main", "legacy", "both"} or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Use main, legacy or both libraries and a limit between 1 and 100")
    for value in (accession, record_id):
        if value is not None and (not isinstance(value, str) or not 1 <= len(value.strip()) <= 200
                                  or any(ord(c) < 32 for c in value)):
            raise ValueError("Use a nonempty accession or record ID of at most 200 characters")
    accession = accession.strip() if accession is not None else None
    if record_id is not None:
        record_id = record_id.strip().casefold()
        if not record_id.startswith(("id:", "accession:")):
            record_id = "id:" + record_id.strip("{}")
    if accession is None and record_id is None:
        raise ValueError("An accession or NRC record ID is required")
    if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 2048):
        raise ValueError("Invalid catalog cursor")
    return accession, record_id


def lookup(path, accession=None, library="both", record_id=None, limit=50, cursor=None):
    accession, record_id = normalize_query(accession, library, record_id, limit, cursor)
    db = connect(path)
    try:
        db.execute("BEGIN")
        if accession is not None and not index_ready(db):
            raise RuntimeError("The catalog lookup index is not ready; run python -m adams_rag.catalog_lookup index")
        snapshot = [dict(r) for r in db.execute("SELECT library,final_count,finished,state FROM libraries ORDER BY library")]
        signature = hashlib.sha256(json.dumps([str(Path(path).resolve()), snapshot, accession.casefold() if accession else None,
                                              library, record_id], sort_keys=True).encode()).hexdigest()
        where, params = [], []
        if accession is not None:
            where.append("accession=? COLLATE NOCASE"); params.append(accession)
        if record_id is not None:
            where.append("record_id=?"); params.append(record_id)
        if library == "both":
            where.append("library IN ('main','legacy')")
        else:
            where.append("library=?"); params.append(library)
        table = "items INDEXED BY " + INDEX if accession is not None else "items"
        total = db.execute("SELECT count(*) FROM " + table + " WHERE " + " AND ".join(where), params).fetchone()[0]
        if cursor:
            try:
                value = json.loads(base64.b64decode(cursor, altchars=b'-_', validate=True))
                after = value["after"]
                if value["query"] != signature or not isinstance(after, list) or len(after) != 2 or not all(isinstance(v, str) for v in after):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                raise ValueError("Invalid cursor or catalog/query changed; restart the lookup") from None
            where.append("(library,record_id)>(?,?)"); params.extend(after)
        rows = db.execute("SELECT * FROM " + table + " WHERE " + " AND ".join(where) +
                          " ORDER BY library,record_id LIMIT ?", (*params, limit + 1)).fetchall()
        records = []
        for row in rows[:limit]:
            metadata = json.loads(zlib.decompress(row["metadata"]))
            for key in ("content", "text", "SearchExcerpt"):
                metadata.pop(key, None)
            records.append({"source_key": f"nrc:{row['library']}:{row['record_id']}",
                            "library": row["library"], "record_id": row["record_id"],
                            "accession": row["accession"], "is_package": bool(row["is_package"]),
                            "metadata": metadata, "content_scope": "catalog_metadata", "full_text_included": False})
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = base64.urlsafe_b64encode(json.dumps({"query": signature,
                "after": [last["library"], last["record_id"]]}, separators=(",", ":")).encode()).decode()
        return {"matched_records": total, "records": records, "next_cursor": next_cursor,
                "snapshot": snapshot, "content_scope": "catalog_metadata", "full_archive_verified": False}
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["index", "lookup"])
    parser.add_argument("accession", nargs="?")
    parser.add_argument("--catalog", default="../adams-catalog/catalog.sqlite")
    parser.add_argument("--library", choices=["main", "legacy", "both"], default="both")
    parser.add_argument("--record-id")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--cursor")
    parser.add_argument("--updates", help="Include reconciled supplements from this directory for lookup")
    args = parser.parse_args(argv)
    if args.action == "index":
        result = build_index(args.catalog)
    elif args.updates:
        from .catalog_collection import lookup_available
        result = lookup_available(args.catalog, args.updates, accession=args.accession, library=args.library,
                                  record_id=args.record_id, limit=args.limit, cursor=args.cursor)
    else:
        result = lookup(args.catalog, args.accession, args.library, args.record_id, args.limit, args.cursor)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
