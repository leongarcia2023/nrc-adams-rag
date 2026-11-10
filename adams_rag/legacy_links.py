"""Map explicit NRC NUDOCS report references to digitized main-library records."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import signal
import sqlite3
import zlib

from .catalog_lookup import connect, lookup as catalog_lookup
from .storage import StoragePause, require_space

PARSER_VERSION = 2
REFERENCE = re.compile(r"\s*NUDOCS\s+(\d{10})\s*", re.I)
LEGACY_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d{10}(?![A-Za-z0-9])")


def default_path(catalog):
    return Path(catalog).resolve().with_name("catalog-legacy-links.sqlite")


def snapshot(db, catalog, parser_version=PARSER_VERSION):
    rows = [dict(r) for r in db.execute("SELECT library,final_count,finished,state FROM libraries ORDER BY library")]
    if not rows or any(r["final_count"] is None or not r["finished"] for r in rows):
        raise ValueError("Legacy linking requires a reconciled catalog")
    return hashlib.sha256(json.dumps([str(Path(catalog).resolve()), rows, parser_version], sort_keys=True).encode()).hexdigest()


def _parse_references(metadata):
    """Accept whole report-number tokens, including NRC's comma-separated fields."""
    values = metadata.get("DocumentReportNumber") or []
    if isinstance(values, str):
        values = [values]
    found, unparsed = {}, []
    for value in values:
        if not isinstance(value, str):
            continue
        rejected = False
        for token in value.split(","):
            match = REFERENCE.fullmatch(token)
            if match:
                found[match[1]] = value
            elif "nudocs" in token.casefold():
                rejected = True
        if rejected:
            unparsed.append(value)
    return found, unparsed


def references(metadata):
    found, unparsed = _parse_references(metadata)
    return found, len(unparsed)


def upgrade(catalog, destination=None, *, cancelled=None, progress=None):
    """Audit a completed v1 map, then atomically add only newly explicit references."""
    destination = Path(destination or default_path(catalog))
    require_space(destination.parent)
    with destination.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A legacy-link build is already running") from None
        source, db = connect(catalog), sqlite3.connect(f"file:{destination.resolve()}?mode=rw", uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            source.execute("BEGIN")
            state = dict(db.execute("SELECT * FROM link_state WHERE id=1").fetchone())
            signature = snapshot(source, catalog)
            if state["snapshot"] == signature:
                return {"state": "already_current", "added_links": 0, "unparsed": state["unparsed"]}
            if state["state"] != "complete" or state["snapshot"] != snapshot(source, catalog, 1):
                raise ValueError("Upgrade requires a completed v1 map of this exact catalog snapshot")
            additions, gaps, scanned, new_count = [], [], 0, 0
            for row in source.execute("SELECT rowid,metadata FROM items NOT INDEXED WHERE library='main' AND is_package=0 ORDER BY rowid"):
                metadata = json.loads(zlib.decompress(row["metadata"]))
                found, unparsed = _parse_references(metadata)
                values = metadata.get("DocumentReportNumber") or []
                if isinstance(values, str):
                    values = [values]
                previous = {m[1] for value in values if isinstance(value, str) and (m := REFERENCE.fullmatch(value))}
                additions.extend((accession, row["rowid"], reference) for accession, reference in found.items() if accession not in previous)
                gaps.extend((row["rowid"], reference) for reference in unparsed)
                scanned += 1
                new_count += len(found)
                if len(additions) + len(gaps) > 10000:
                    raise ValueError("Upgrade delta exceeds its memory allowance; build a separate v2 map")
                if scanned % 1000 == 0:
                    require_space(destination.parent)
                    if cancelled and cancelled():
                        raise InterruptedError("Legacy-map upgrade stopped; old map retained")
                    if progress:
                        progress({"scanned": scanned, "added_links": len(additions), "unparsed": len(gaps)})
            if scanned != state["processed"] or new_count != state["links"] + len(additions):
                raise ValueError("Legacy-map reconciliation failed; old map retained")
            require_space(destination.parent)
            if cancelled and cancelled():
                raise InterruptedError("Legacy-map upgrade stopped; old map retained")
            with db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("CREATE TABLE IF NOT EXISTS legacy_link_exceptions(main_row INTEGER NOT NULL,reference TEXT NOT NULL,PRIMARY KEY(main_row,reference)) WITHOUT ROWID")
                db.executemany("INSERT INTO legacy_links VALUES(?,?,?)", additions)
                db.executemany("INSERT OR IGNORE INTO legacy_link_exceptions VALUES(?,?)", gaps)
                db.execute("UPDATE link_state SET snapshot=?,links=?,unparsed=? WHERE id=1", (signature, new_count, len(gaps)))
            return {"state": "upgraded", "scanned": scanned, "added_links": len(additions), "links": new_count,
                    "unparsed": len(gaps), "parser_version": PARSER_VERSION, "full_text_acquired": False}
        finally:
            source.close()
            db.close()


def build(catalog, destination=None, *, max_rows=None, batch_size=1000, min_free_gb=None,
          cancelled=None, progress=None):
    if not 1 <= batch_size <= 10000 or (max_rows is not None and max_rows < 1):
        raise ValueError("Use a positive row budget and batches between 1 and 10,000")
    destination = Path(destination or default_path(catalog))
    if destination.resolve() == Path(catalog).resolve():
        raise ValueError("The derived link database must be separate from the catalog")
    require_space(destination.parent, min_free_gb)
    with destination.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A legacy-link build is already running") from None
        source, db = connect(catalog), sqlite3.connect(destination, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            source.execute("BEGIN")
            signature = snapshot(source, catalog)
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""CREATE TABLE IF NOT EXISTS link_state(
                id INTEGER PRIMARY KEY CHECK(id=1), snapshot TEXT NOT NULL,
                cursor INTEGER NOT NULL DEFAULT 0, processed INTEGER NOT NULL DEFAULT 0,
                links INTEGER NOT NULL DEFAULT 0, unparsed INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'running');
                CREATE TABLE IF NOT EXISTS legacy_links(
                legacy_accession TEXT NOT NULL, main_row INTEGER NOT NULL,
                reference TEXT NOT NULL, PRIMARY KEY(legacy_accession,main_row)) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS legacy_link_exceptions(main_row INTEGER NOT NULL,
                reference TEXT NOT NULL,PRIMARY KEY(main_row,reference)) WITHOUT ROWID;
            """)
            with db:
                db.execute("INSERT OR IGNORE INTO link_state(id,snapshot) VALUES(1,?)", (signature,))
            state = dict(db.execute("SELECT * FROM link_state WHERE id=1").fetchone())
            if state["snapshot"] != signature:
                raise ValueError("Catalog snapshot changed; build a separate link database for the new snapshot")
            processed = 0
            while state["state"] != "complete":
                if cancelled and cancelled():
                    state["state"] = "stopped"
                    break
                if max_rows is not None and processed >= max_rows:
                    state["state"] = "batch_limit"
                    break
                try:
                    require_space(destination.parent, min_free_gb)
                except StoragePause:
                    state["state"] = "storage_pause"
                    break
                count = min(batch_size, max_rows - processed) if max_rows is not None else batch_size
                rows = source.execute("SELECT rowid,metadata FROM items NOT INDEXED WHERE library='main' AND is_package=0 AND rowid>? ORDER BY rowid LIMIT ?",
                                      (state["cursor"], count)).fetchall()
                if not rows:
                    state["state"] = "complete"
                    break
                links, unparsed, gaps = [], 0, []
                for row in rows:
                    found, skipped = _parse_references(json.loads(zlib.decompress(row["metadata"])))
                    links.extend((accession, row["rowid"], reference) for accession, reference in found.items())
                    unparsed += len(skipped)
                    gaps.extend((row["rowid"], reference) for reference in skipped)
                if cancelled and cancelled():
                    state["state"] = "stopped"
                    break
                try:
                    require_space(destination.parent, min_free_gb,
                                  extra_bytes=max(32 * 1024 * 1024, len(links) * 1024))
                except StoragePause:
                    state["state"] = "storage_pause"
                    break
                with db:
                    db.executemany("INSERT INTO legacy_links VALUES(?,?,?)", links)
                    db.executemany("INSERT OR IGNORE INTO legacy_link_exceptions VALUES(?,?)", gaps)
                    db.execute("UPDATE link_state SET cursor=?,processed=processed+?,links=links+?,unparsed=unparsed+?,state='running' WHERE id=1",
                               (rows[-1]["rowid"], len(rows), len(links), unparsed))
                processed += len(rows)
                state = dict(db.execute("SELECT * FROM link_state WHERE id=1").fetchone())
                if progress:
                    progress(state.copy())
            with db:
                db.execute("UPDATE link_state SET state=? WHERE id=1", (state["state"],))
            return {**state, "processed_this_run": processed, "full_text_acquired": False}
        finally:
            source.close()
            db.close()


def lookup(catalog, accession, destination=None, limit=20):
    if not isinstance(accession, str) or not re.fullmatch(r"\d{10}", accession):
        raise ValueError("Use a ten-digit legacy accession")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Use a limit between 1 and 100")
    legacy = catalog_lookup(catalog, accession, library="legacy", limit=100)
    result = {"legacy_accession": accession, "legacy_matched_records": legacy["matched_records"],
              "legacy_source_keys": [r["source_key"] for r in legacy["records"]],
              "legacy_records_truncated": bool(legacy["next_cursor"]), "records": [],
              "matched_records": 0, "truncated": False, "state": "not_built",
              "relationship": "nrc_report_number_reference", "identity_equivalence_verified": False,
              "full_text_included": False, "content_scope": "catalog_metadata"}
    path = Path(destination or default_path(catalog))
    if not path.is_file():
        return result
    source, db = connect(catalog), connect(path)
    try:
        source.execute("BEGIN")
        db.execute("BEGIN")
        state = db.execute("SELECT * FROM link_state WHERE id=1").fetchone()
        if not state or state["snapshot"] != snapshot(source, catalog):
            result["state"] = "stale"
            return result
        result["state"] = state["state"]
        result["processed_main_records"] = state["processed"]
        result["unparsed_references"] = state["unparsed"]
        result["matched_records"] = db.execute("SELECT count(*) FROM legacy_links WHERE legacy_accession=?", (accession,)).fetchone()[0]
        result["truncated"] = result["matched_records"] > limit
        for link in db.execute("SELECT main_row,reference FROM legacy_links WHERE legacy_accession=? ORDER BY main_row LIMIT ?", (accession, limit)):
            row = source.execute("SELECT * FROM items WHERE rowid=? AND library='main' AND is_package=0", (link["main_row"],)).fetchone()
            if row is None:
                raise ValueError("Linked catalog record is missing; rebuild against the current snapshot")
            metadata = json.loads(zlib.decompress(row["metadata"]))
            if accession not in references(metadata)[0]:
                raise ValueError("Linked catalog reference changed; rebuild against the current snapshot")
            result["records"].append({"accession": row["accession"], "record_id": row["record_id"],
                "source_key": f"nrc:main:{row['record_id']}", "title": metadata.get("DocumentTitle"),
                "document_date": metadata.get("DocumentDate"), "reference": link["reference"]})
        return result
    finally:
        source.close()
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "lookup", "upgrade"])
    parser.add_argument("accession", nargs="?")
    parser.add_argument("--catalog", default="../adams-catalog/catalog.sqlite")
    parser.add_argument("--destination")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    if args.action == "build":
        result = build(args.catalog, args.destination, max_rows=args.max_rows, cancelled=lambda: stopping[0])
    elif args.action == "upgrade":
        result = upgrade(args.catalog, args.destination, cancelled=lambda: stopping[0])
    else:
        result = lookup(args.catalog, args.accession, args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
