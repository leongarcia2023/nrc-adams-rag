"""Resumable, metadata-only inventory of both public ADAMS libraries.

Accession-prefix partitions include undated records. Each split has an explicit
complement, so no alphabet or accession-number pattern is assumed. A partition
is accepted only when its unique row count agrees with the API's count.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import time
import zlib
from pathlib import Path

from .client import AdamsClient, BudgetStop, TemporaryRequestError, unpack_search
from .config import load_api_key
from .store import Store
from .metrics import install_metrics
from .transport import PersistentOpener
from .prefetch import PrefetchClient
from .identity import record_id


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def number(value):
    if value is None or value == "":
        return None, "missing"
    if isinstance(value, bool):
        return None, "invalid"
    text = str(value).strip()
    if not text.isascii() or not text.isdigit():
        return None, "invalid"
    n = int(text)
    if n > 2**63 - 1:
        return None, "invalid"
    return n, "positive" if n else "zero"


def filters_for(prefix, excluded, exact):
    if exact:
        return [{"field": "AccessionNumber", "operator": "equals", "value": prefix}]
    filters = ([{"field": "AccessionNumber", "operator": "starts", "value": prefix}]
               if prefix else [])
    for item in excluded:
        suffix = item["value"] if isinstance(item, dict) else item
        is_exact = item.get("exact", False) if isinstance(item, dict) else suffix == ""
        filters.append({"field": "AccessionNumber",
                        "operator": "notequals" if is_exact else "notstarts",
                        "value": prefix + suffix})
    return filters


def matches(accession, prefix, excluded, exact):
    accession, prefix = accession.casefold(), prefix.casefold()
    if exact:
        return accession == prefix
    if not accession.startswith(prefix):
        return False
    for item in excluded:
        suffix = item["value"] if isinstance(item, dict) else item
        is_exact = item.get("exact", False) if isinstance(item, dict) else suffix == ""
        value = prefix + suffix.casefold()
        if (accession == value) if is_exact else accession.startswith(value):
            return False
    return True


class Catalog:
    def __init__(self, directory):
        from .offload_guard import require_local_data
        require_local_data(Path(directory) / 'catalog.sqlite')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "catalog.sqlite"
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS libraries(
            library TEXT PRIMARY KEY, initial_count INTEGER, final_count INTEGER,
            started TEXT, finished TEXT, state TEXT DEFAULT 'pending');
          CREATE TABLE IF NOT EXISTS partitions(
            id INTEGER PRIMARY KEY, parent INTEGER REFERENCES partitions(id),
            library TEXT NOT NULL, prefix TEXT NOT NULL, excluded TEXT NOT NULL,
            exact INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending',
            expected INTEGER, offset INTEGER NOT NULL DEFAULT 0,
            last_accession TEXT, note TEXT, updated TEXT);
          CREATE INDEX IF NOT EXISTS partition_work ON partitions(state,library,id);
          CREATE TABLE IF NOT EXISTS items(
            library TEXT NOT NULL, record_id TEXT NOT NULL, accession TEXT NOT NULL, partition_id INTEGER NOT NULL,
            size_bytes INTEGER, size_state TEXT NOT NULL, pages INTEGER,
            is_package INTEGER NOT NULL, url TEXT, metadata BLOB NOT NULL,
            PRIMARY KEY(library,record_id));
          CREATE INDEX IF NOT EXISTS item_partition ON items(partition_id);
          CREATE TABLE IF NOT EXISTS pages(
            partition_id INTEGER NOT NULL, offset INTEGER NOT NULL, count INTEGER NOT NULL,
            returned INTEGER NOT NULL, ids_sha256 TEXT NOT NULL, recorded TEXT NOT NULL,
            PRIMARY KEY(partition_id,offset));
          CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY, recorded TEXT NOT NULL, event TEXT NOT NULL, detail TEXT);
          CREATE TABLE IF NOT EXISTS audit_queue(partition_id INTEGER PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS leaf_restarts(partition_id INTEGER PRIMARY KEY, attempts INTEGER NOT NULL);
        """)
        if "record_id" not in {r[1] for r in self.db.execute("PRAGMA table_info(items)")}:
            # Preserve the early inventory when migrating from accession keys:
            # distinct legacy NRC record IDs can share one accession number.
            with self.db:
                self.db.execute("ALTER TABLE items RENAME TO items_v1")
                self.db.execute("""CREATE TABLE items(
                  library TEXT NOT NULL,record_id TEXT NOT NULL,accession TEXT NOT NULL,partition_id INTEGER NOT NULL,
                  size_bytes INTEGER,size_state TEXT NOT NULL,pages INTEGER,is_package INTEGER NOT NULL,
                  url TEXT,metadata BLOB NOT NULL,PRIMARY KEY(library,record_id))""")
                for r in self.db.execute("SELECT * FROM items_v1"):
                    doc = json.loads(zlib.decompress(r["metadata"]))
                    self.db.execute("INSERT INTO items VALUES(?,?,?,?,?,?,?,?,?,?)", (
                        r["library"],record_id(doc),r["accession"],r["partition_id"],r["size_bytes"],
                        r["size_state"],r["pages"],r["is_package"],r["url"],r["metadata"]))
                self.db.execute("DROP TABLE items_v1")
                self.db.execute("CREATE INDEX item_partition ON items(partition_id)")
        with self.db:
            for library in ("main", "legacy"):
                if not self.db.execute("SELECT 1 FROM libraries WHERE library=?", (library,)).fetchone():
                    self.db.execute("INSERT INTO libraries(library) VALUES(?)", (library,))
                    self.db.execute("INSERT INTO partitions(library,prefix,excluded) VALUES(?,'','[]')", (library,))
        install_metrics(self.db)

    def event(self, event, detail):
        with self.db:
            self.db.execute("INSERT INTO events(recorded,event,detail) VALUES(?,?,?)", (now(), event, detail))

    def fail(self, partition, detail):
        with self.db:
            self.db.execute("UPDATE partitions SET state='blocked',note=?,updated=? WHERE id=?",
                            (detail, now(), partition["id"]))
        raise RuntimeError(detail)

    def next_partition(self):
        # Round-robin libraries; depth-first within each library completes sections early.
        last = self.db.execute("SELECT library FROM partitions WHERE updated IS NOT NULL ORDER BY updated DESC LIMIT 1").fetchone()
        previous = last[0] if last else "legacy"
        return self.db.execute("""SELECT * FROM partitions WHERE state IN ('pending','running')
          ORDER BY (library=?) ASC,length(prefix) DESC,id DESC LIMIT 1""", (previous,)).fetchone()

    def reset_leaf(self, p):
        with self.db:
            self.db.execute("DELETE FROM items WHERE partition_id=?", (p["id"],))
            self.db.execute("DELETE FROM pages WHERE partition_id=?", (p["id"],))
            self.db.execute("UPDATE partitions SET expected=NULL,offset=0,state='pending',last_accession=NULL,note=NULL,updated=? WHERE id=?",
                            (now(), p["id"]))

    def audit_step(self, client):
        p = self.db.execute("""SELECT p.* FROM partitions p JOIN audit_queue a ON a.partition_id=p.id
          ORDER BY length(p.prefix),p.id LIMIT 1""").fetchone()
        if p is None:
            return False
        response = getattr(client, "search_count", client.search)(library=p["library"],
                                 filters=filters_for(p["prefix"], json.loads(p["excluded"]), p["exact"]))
        docs = unpack_search(response)
        count = response.get("count")
        if type(count) is not int or count < 0 or count < len(docs):
            raise RuntimeError("Invalid count during catalog reconciliation")
        if any(not matches(d["AccessionNumber"],p["prefix"],json.loads(p["excluded"]),p["exact"]) for d in docs):
            raise RuntimeError("Out-of-partition response during catalog reconciliation")
        if count != p["expected"]:
            if p["state"] == "split":
                with self.db:
                    self.db.execute("UPDATE partitions SET expected=?,updated=? WHERE id=?", (count,now(),p["id"]))
                    self.db.execute("INSERT OR IGNORE INTO audit_queue SELECT id FROM partitions WHERE parent=?", (p["id"],))
            else:
                self.reset_leaf(p)
            self.event("count_reconciliation", f"Partition {p['id']}: {p['expected']} -> {count}")
        with self.db:
            self.db.execute("DELETE FROM audit_queue WHERE partition_id=?", (p["id"],))
        return True

    def step(self, client, max_leaf=1000):
        if self.audit_step(client):
            return True
        p = self.next_partition()
        if p is None:
            return False
        excluded = json.loads(p["excluded"])
        response = client.search(library=p["library"], skip=p["offset"],
                                 filters=filters_for(p["prefix"], excluded, p["exact"]))
        docs = unpack_search(response)
        count = response.get("count")
        if type(count) is not int or count < 0:
            self.fail(p, "Missing or invalid API count; coverage cannot be reconciled")
        ids = [d["AccessionNumber"] for d in docs]
        record_ids = [record_id(d) for d in docs]
        if len(set(record_ids)) != len(record_ids):
            self.fail(p, "Duplicate NRC record IDs within an API page")
        if any(not matches(x, p["prefix"], excluded, p["exact"]) for x in ids):
            self.fail(p, "API returned accessions outside the requested partition")
        # The API sorts accession values case-sensitively: e.g. a lowercase
        # ml100040062 follows uppercase ML100040600. Filters remain caseless.
        if ids != sorted(ids):
            self.fail(p, "API accession sort is unstable or differs from expected ordering")
        if p["last_accession"] and ids and ids[0] < p["last_accession"]:
            self.fail(p, "Repeated or overlapping API page; offset was not advanced")
        if p["expected"] is not None and count != p["expected"]:
            row = self.db.execute("SELECT attempts FROM leaf_restarts WHERE partition_id=?", (p["id"],)).fetchone()
            attempts = (row[0] if row else 0) + 1
            if attempts > 3:
                self.fail(p, "Partition repeatedly changed during enumeration; review required")
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO leaf_restarts VALUES(?,?)", (p["id"],attempts))
            self.reset_leaf(p)
            self.event("leaf_restarted", f"Partition {p['id']}: count changed; safely restarting this leaf")
            return True
        if count < p["offset"] + len(docs) or (not docs and p["offset"] < count):
            self.fail(p, "API count disagrees with returned records; possible truncation")
        if count <= max_leaf and any(self.db.execute("SELECT 1 FROM items WHERE library=? AND record_id=?",
                                                    (p["library"], rid)).fetchone() for rid in record_ids):
            self.fail(p, "Repeated NRC record ID across pages or partitions")
        stamp = now()
        with self.db:
            if p["parent"] is None:
                self.db.execute("UPDATE libraries SET initial_count=COALESCE(initial_count,?),started=COALESCE(started,?),state='running' WHERE library=?",
                                (count, stamp, p["library"]))
            self.db.execute("UPDATE partitions SET expected=?,updated=? WHERE id=?", (count, stamp, p["id"]))
            self.db.execute("INSERT INTO pages VALUES(?,?,?,?,?,?)", (
                p["id"], p["offset"], count, len(ids), hashlib.sha256(json.dumps(record_ids).encode()).hexdigest(), stamp))
            if count > max_leaf:
                if p["offset"] or p["exact"] or not docs:
                    raise RuntimeError("Cannot subdivide this partition safely")
                # Children for every next character observed, and an explicit complement.
                # The live API silently ignores one-character text filters.
                # Start with at least two characters; then subdivide one at a time.
                cut = max(2, len(p["prefix"]) + 1)
                # Case-insensitive filters must not create overlapping children
                # for differently capitalized versions of the same prefix.
                suffixes = sorted({(x[len(p["prefix"]):cut].casefold(), len(x) < cut) for x in ids})
                for suffix, is_exact in suffixes:
                    self.db.execute("INSERT INTO partitions(parent,library,prefix,excluded,exact) VALUES(?,?,?,?,?)",
                                    (p["id"], p["library"], p["prefix"] + suffix, "[]", int(is_exact)))
                exclusions = [{"value": suffix, "exact": True} if is_exact else suffix for suffix, is_exact in suffixes]
                self.db.execute("INSERT INTO partitions(parent,library,prefix,excluded) VALUES(?,?,?,?)",
                                (p["id"], p["library"], p["prefix"], json.dumps(excluded + exclusions)))
                self.db.execute("UPDATE partitions SET state='split' WHERE id=?", (p["id"],))
                return True
            for doc in docs:
                size, size_state = number(doc.get("DocumentSize"))
                pages, _ = number(doc.get("EstimatedPageCount"))
                metadata = {k: v for k, v in doc.items() if k not in {"content", "text"}}
                packed = zlib.compress(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode())
                self.db.execute("INSERT INTO items VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    p["library"], record_id(doc), doc["AccessionNumber"], p["id"], size, size_state, pages,
                    int(str(doc.get("IsPackage", "")).lower() == "yes" or doc.get("ItemType") == "package"),
                    doc.get("Url"), packed))
            offset = p["offset"] + len(docs)
            finished = offset == count
            self.db.execute("UPDATE partitions SET offset=?,state=?,last_accession=? WHERE id=?",
                            (offset, "count_matched" if finished else "running", ids[-1] if ids else p["last_accession"], p["id"]))
        return True

    def reconcile(self, client):
        if self.next_partition() is not None or self.db.execute("SELECT 1 FROM audit_queue LIMIT 1").fetchone():
            return False
        if self.db.execute("SELECT 1 FROM partitions WHERE state='blocked' LIMIT 1").fetchone():
            return False
        mismatches = []
        for parent in self.db.execute("SELECT id,expected FROM partitions WHERE state='split'"):
            children = self.db.execute("SELECT expected FROM partitions WHERE parent=?", (parent["id"],)).fetchall()
            if not children or any(c[0] is None for c in children) or sum(c[0] for c in children) != parent["expected"]:
                mismatches.append(parent["id"])
        if mismatches:
            # Revisit inconsistent ancestors and their immediate children;
            # unchanged historical branches do not need another full crawl.
            with self.db:
                for partition_id in mismatches:
                    self.db.execute("INSERT OR IGNORE INTO audit_queue SELECT id FROM partitions WHERE id=? OR parent=?",
                                    (partition_id,partition_id))
            self.event("reconciliation_requested", "Child counts changed; auditing partition tree")
            return False
        changed = False
        for lib in self.db.execute("SELECT * FROM libraries").fetchall():
            if lib["state"] == "enumerated_count_matched":
                continue
            response = getattr(client, "search_count", client.search)(library=lib["library"])
            unpack_search(response)
            count = response.get("count")
            observed = self.db.execute("SELECT count(*) FROM items WHERE library=?", (lib["library"],)).fetchone()[0]
            if type(count) is not int or count < 0:
                raise RuntimeError("Invalid final library count")
            if count != observed:
                with self.db:
                    self.db.execute("INSERT OR IGNORE INTO audit_queue SELECT id FROM partitions WHERE library=? AND parent IS NULL", (lib["library"],))
                changed = True
                continue
            with self.db:
                self.db.execute("UPDATE libraries SET final_count=?,finished=?,state='enumerated_count_matched' WHERE library=?",
                                (count, now(), lib["library"]))
        return not changed

    def summary(self):
        libraries = [dict(r) for r in self.db.execute("SELECT * FROM libraries ORDER BY library")]
        counts = dict(self.db.execute("SELECT state,count(*) FROM partitions GROUP BY state"))
        metrics = [dict(r) for r in self.db.execute("SELECT * FROM inventory_metrics WHERE records>0 ORDER BY library")]
        return {"updated_utc": now(), "libraries": libraries, "partitions": counts, "inventory": metrics,
                "catalog_records": sum(r["records"] for r in metrics),
                "pending_audits": self.db.execute("SELECT count(*) FROM audit_queue").fetchone()[0],
                "reported_library_entries": sum(r["initial_count"] or 0 for r in libraries),
                "enumeration_count_matched": all(r["state"] == "enumerated_count_matched" for r in libraries),
                "free_disk_bytes": shutil.disk_usage(self.directory).free,
                "notes": ["Metadata only; no PDFs or full text downloaded by this inventory.",
                          "Partial totals are not whole-archive storage estimates.",
                          "Libraries are counted separately; cross-library accessions and shared file URLs may overlap.",
                          "DocumentSize is API-reported; actual file lengths have not been verified.",
                          "Count reconciliation is not an authoritative point-in-time NRC snapshot."]}

    def report(self, state, detail=""):
        report = self.summary()
        report.update(worker_state=state, worker_detail=detail, worker_pid=os.getpid())
        if getattr(self, "client", None) is not None:
            report["effective_requests_per_second"] = 1 / self.client.gate.interval
            if isinstance(self.client, PrefetchClient):
                report["prefetch"] = {**self.client.stats, "maximum_concurrent_fetches": 2,
                                      "cached_or_pending_pages": len(self.client.pending)}
        if getattr(self,"response_db",None) is not None:
            first = self.response_db.execute("SELECT min(at) FROM http_responses WHERE scope=?", (self.response_scope,)).fetchone()[0]
            counts = self.response_db.execute("SELECT status,count(*) FROM http_responses WHERE scope=? GROUP BY status", (self.response_scope,)).fetchall()
            report["api_response_tracking"] = {
                "started_utc": dt.datetime.fromtimestamp(first,dt.timezone.utc).isoformat() if first else None,
                "response_codes": {str(code) if code else "network_error": count for code,count in counts},
                "note": "Earlier HTTP response codes were not retained; these counts cover only the tracking interval."}
        atomic_write(self.directory / "status.json", json.dumps(report, indent=2))
        lines = ["NRC ADAMS public catalog inventory", "", f"Updated: {report['updated_utc']}",
                 f"Worker: {state}", detail, "",
                 f"Records cataloged: {report['catalog_records']:,}",
                 f"Library entries reported by API: {report['reported_library_entries']:,}",
                 f"Enumeration reconciled: {report['enumeration_count_matched']}", ""]
        for r in report["inventory"]:
            lines.append(f"{r['library']}: {r['records']:,} records; {r['nonpackage_reported_bytes'] / 1e12:.6f} TB of reported non-package sizes so far; {r['missing_sizes']:,} missing sizes.")
        if "api_response_tracking" in report:
            tracking=report["api_response_tracking"]
            lines += ["", f"HTTP response tracking since: {tracking['started_utc'] or 'awaiting first response'}",
                      f"Recorded response codes: {json.dumps(tracking['response_codes'])}", tracking["note"]]
        lines += ["", *report["notes"]]
        atomic_write(self.directory / "STATUS.txt", "\n".join(lines) + "\n")
        return report

    def export(self):
        temporary = self.directory / "catalog.jsonl.gz.partial"
        with gzip.open(temporary, "wt", encoding="utf-8") as file:
            for row in self.db.execute("SELECT library,metadata FROM items ORDER BY library,accession"):
                file.write(json.dumps({"library": row[0], "document": json.loads(zlib.decompress(row[1]))}, ensure_ascii=False) + "\n")
        temporary.replace(self.directory / "catalog.jsonl.gz")


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def wait_until(catalog, resume, clock=time.time, sleep=time.sleep):
    """Wait without issuing requests; honor an operator stop during cooldown."""
    while clock() < resume:
        if (catalog.directory / "STOP").exists():
            catalog.report("stopped", "STOP file found during cooldown.")
            return False
        sleep(min(20, max(0, resume - clock())))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--rate-db", required=True, help="Share the existing API request ledger")
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--daily-limit", type=int, default=5000, help="Local budget, not an NRC-published quota")
    parser.add_argument("--rps", type=float, default=0.2)
    parser.add_argument("--min-free-gb", type=float)
    parser.add_argument("--max-leaf", type=int, default=1000)
    parser.add_argument("--continuous", action="store_true", help="Wait for saved cooldowns and local daily budget resets")
    parser.add_argument("--prefetch", action="store_true", help="Overlap at most two page fetches under the same global request limit")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    from .storage import reserve_gb
    try:
        args.min_free_gb = reserve_gb(args.min_free_gb)
    except ValueError as error:
        parser.error(str(error))
    if args.requests < 1 or not 1 <= args.max_leaf <= 1000:
        parser.error("Invalid inventory bounds")
    catalog = Catalog(args.directory)
    if args.status:
        print(json.dumps(catalog.summary(), indent=2))
        catalog.db.close()
        return
    lock = (catalog.directory / "worker.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An inventory worker already holds this catalog lock")
    atomic_write(catalog.directory / "worker.pid", str(os.getpid()) + "\n")
    rate_store = Store(args.rate_db)
    transport = None
    if args.prefetch:
        client = PrefetchClient(rate_store, load_api_key(), args.requests, args.rps, args.daily_limit, args.max_leaf)
    else:
        transport = PersistentOpener()
        client = AdamsClient(rate_store, load_api_key(), args.requests, args.rps, args.daily_limit,
                             opener=transport, include_search_content=False)
    catalog.client = client
    catalog.response_db, catalog.response_scope = rate_store.db, client.scope
    last_report = 0
    transient_failures = 0
    try:
        catalog.report("running", "Conservative local rate and daily budget; actual NRC quota unverified.")
        while True:
            if (catalog.directory / "STOP").exists():
                catalog.report("stopped", "STOP file found; remove it before resuming.")
                return
            if shutil.disk_usage(catalog.directory).free < args.min_free_gb * 1e9:
                catalog.report("needs_storage", f"Stopped to preserve at least {args.min_free_gb:g} GB free. Move the catalog to a larger drive and resume.")
                return
            try:
                progressed = catalog.step(client, args.max_leaf)
                transient_failures = 0
                if not progressed:
                    if catalog.reconcile(client):
                        packed_bytes = catalog.db.execute("SELECT coalesce(sum(length(metadata)),0) FROM items").fetchone()[0]
                        if shutil.disk_usage(catalog.directory).free > 2 * packed_bytes + args.min_free_gb * 1e9:
                            catalog.export()
                            detail = "Both library counts and every partition reconciled. SQLite catalog and compressed JSONL export are ready."
                        else:
                            detail = "Both library counts and every partition reconciled. SQLite catalog is ready; compressed JSONL export deferred to preserve disk space."
                        catalog.report("finished", detail)
                    else:
                        if catalog.db.execute("SELECT 1 FROM audit_queue LIMIT 1").fetchone():
                            catalog.report("reconciling", "Checking count changes and refreshing affected sections.")
                            continue
                        catalog.report("needs_review", "Blocked partitions require inspection before resuming.")
                    return
            except TemporaryRequestError as error:
                if not args.continuous:
                    raise
                transient_failures = min(5, transient_failures + 1)
                delay = min(3600, 300 * 2 ** (transient_failures - 1))
                client.gate.defer(delay)
                resume = rate_store.db.execute("SELECT next_at FROM http_state WHERE scope=?", (client.gate.scope,)).fetchone()[0]
                detail = f"{error}. Retrying automatically after {dt.datetime.fromtimestamp(resume,dt.timezone.utc).isoformat()}."
                catalog.event("temporary_request_wait",detail)
                catalog.report("connection_wait",detail)
                print(json.dumps({"time":now(),"state":"connection_wait","retry_after_seconds":delay}),flush=True)
                if not wait_until(catalog,resume):
                    return
                continue
            except BudgetStop as error:
                if not args.continuous:
                    catalog.report("budget_wait", str(error))
                    return
                stamp = time.time()
                next_at = rate_store.db.execute("SELECT next_at FROM http_state WHERE scope=?", (client.gate.scope,)).fetchone()[0]
                recent = rate_store.db.execute("SELECT min(at),count(*) FROM http_requests WHERE scope=? AND at>?", (client.gate.scope, stamp - 86400)).fetchone()
                resume = max(stamp + 10, next_at)
                if recent[1] >= args.daily_limit:
                    resume = max(resume, recent[0] + 86401)
                if client.remaining <= 0:
                    client.remaining = args.requests
                catalog.report("budget_wait", f"{error}. Next eligible request after {dt.datetime.fromtimestamp(resume,dt.timezone.utc).isoformat()}.")
                if not wait_until(catalog,resume):
                    return
                catalog.report("running", "Resuming after persisted cooldown or local budget reset.")
            if time.monotonic() - last_report > 60:
                result = catalog.report("running", "Metadata inventory in progress; partial byte totals are not archive estimates.")
                print(json.dumps({"time": now(), "records": result["catalog_records"], "partitions": result["partitions"]}), flush=True)
                last_report = time.monotonic()
    except KeyboardInterrupt:
        catalog.report("stopped", "Interrupted; committed pages are safe to resume.")
    except Exception as error:
        # No request bodies or credential-bearing exceptions enter the log.
        message = str(error) if isinstance(error, (RuntimeError, ValueError, sqlite3.Error)) else type(error).__name__
        catalog.event("worker_error", message)
        catalog.report("needs_review", message)
        print(json.dumps({"error": message}), flush=True)
        raise SystemExit(1)
    finally:
        if args.prefetch:
            client.close()
        elif transport is not None:
            transport.close()
        rate_store.db.close()
        catalog.db.close()
        lock.close()


if __name__ == "__main__":
    main()
