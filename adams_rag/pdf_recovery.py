"""Recover missing main-library text from official PDFs, with durable retry outcomes."""
import argparse
import datetime as dt
import fcntl
import hashlib
import heapq
import http.client
import json
import os
from pathlib import Path
import re
import signal
import shutil
import tempfile
import time
import traceback
import urllib.error
import urllib.request

from .client import BudgetStop, retry_seconds
from .catalog import number as catalog_number
from .ingest import PDFExtractionPending, PDFImportLimit, import_pdf
from .storage import StoragePause, require_space, reserve_gb
from .pdf_policy import DEFAULT_PDF_BYTES, file_sha256, validate_pdf_bytes

MAX_BYTES = DEFAULT_PDF_BYTES
ACCESSION = re.compile(r"ML[0-9A-Z]{8,12}\Z")

# Drive selection from the existing status index and the much smaller recovery
# ledger. The disjoint branches retain changed-policy gaps without scanning all
# indexed document bodies or deduplicating their metadata in a temporary table.
RECOVERY_CANDIDATES_SQL = """WITH eligible(queue_row) AS (
    SELECT rowid FROM documents WHERE status IN ('missing_text','unavailable_text')
    UNION ALL
    SELECT d.rowid FROM pdf_recovery r JOIN documents d USING(accession)
    WHERE r.outcome='indexed_with_gaps' AND r.policy<>?
      AND d.status NOT IN ('missing_text','unavailable_text')
)
SELECT d.rowid AS queue_row,d.accession,d.metadata,p.sha256
FROM eligible e CROSS JOIN documents d ON d.rowid=e.queue_row
LEFT JOIN pdf_recovery p ON p.accession=d.accession
WHERE p.accession IS NULL OR p.next_retry<=? OR
      (p.outcome IN ('needs_review','no_readable_text','indexed_with_gaps') AND p.policy<>?)"""


class RecoveryReview(RuntimeError):
    """A source requires review; do not retry automatically without a changed policy."""


def extraction_failure(error, progress):
    # Keep the failing code location and progress, without arbitrary exception
    # messages that may include source content or connection details.
    return json.dumps({"error_type": type(error).__name__, "last_progress": progress,
                       "changed_fields": getattr(error,"changed_fields",None),
                       "frames": [{"file": Path(frame.filename).name,
                                   "line": frame.lineno, "function": frame.name}
                                  for frame in traceback.extract_tb(error.__traceback__)[-12:]]})


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def official_url(accession, metadata):
    if not ACCESSION.fullmatch(accession):
        raise RecoveryReview("Unsupported main-library accession")
    url = f"https://www.nrc.gov/docs/{accession[:6]}/{accession}.pdf"
    if metadata.get("Url") != url:
        raise RecoveryReview("Catalog URL does not match the expected official accession PDF")
    return url


def download_pdf(url, target, client, *, opener=None, cancelled=None, max_bytes=MAX_BYTES, progress=None):
    """No subscription key, redirects, range acceleration or automatic retries."""
    validate_pdf_bytes(max_bytes)
    opener = opener or urllib.request.build_opener(NoRedirects()).open
    require_space(target.parent, extra_bytes=max_bytes + 32 * 1024 * 1024)
    if cancelled and cancelled():
        raise InterruptedError("PDF recovery stopped")
    client.gate.wait(reserve=client.reserve)
    request = urllib.request.Request(url, headers={"Accept": "application/pdf", "User-Agent": "adams-rag-prototype/0.2"})
    started = time.monotonic()
    time_limit = max(120, min(900, max_bytes / (512 * 1024)))
    last_report = started
    temporary = None
    try:
        with opener(request, timeout=30) as response:
            status = getattr(response, "status", 200)
            client.record_response(status, "pdf_response", started, headers=response.headers)
            if status != 200:
                raise RecoveryReview("PDF endpoint did not return a complete HTTP 200 response")
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    length = int(length)
                except ValueError:
                    raise RecoveryReview("Invalid PDF Content-Length") from None
                if not 0 <= length <= max_bytes:
                    raise RecoveryReview(f"PDF exceeds the {max_bytes / 1024**2:g} MiB recovery allowance (reported {length} bytes)")
            count = 0
            checksum = hashlib.sha256()
            def report():
                if progress:
                    progress({"phase": "downloading", "bytes_received": count,
                              "reported_bytes": length, "max_bytes": max_bytes,
                              "time_limit_seconds": time_limit, "partial_text_published": False})
            report()
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as output:
                temporary = Path(output.name)
                while True:
                    if cancelled and cancelled():
                        raise InterruptedError("PDF recovery stopped")
                    require_space(target.parent, extra_bytes=max_bytes + 32 * 1024 * 1024)
                    now = time.monotonic()
                    if now - started > time_limit:
                        raise TimeoutError("PDF download time allowance reached")
                    block = response.read(min(256 * 1024, max_bytes + 1 - count))
                    if not block:
                        break
                    count += len(block)
                    if count > max_bytes:
                        raise RecoveryReview(f"PDF exceeds the {max_bytes / 1024**2:g} MiB recovery allowance")
                    checksum.update(block)
                    output.write(block)
                    if now - last_report >= 1:
                        report()
                        last_report = now
            report()
            if length is not None and count != length:
                raise RecoveryReview("Incomplete PDF response")
            with temporary.open("rb") as source:
                if not source.read(1024).lstrip().startswith(b"%PDF-"):
                    raise RecoveryReview("Response is not a PDF")
            require_space(target.parent)
            os.replace(temporary, target)
            return {"sha256": checksum.hexdigest(), "bytes": count}
    except urllib.error.HTTPError as error:
        delay = retry_seconds(error.headers.get("Retry-After"), time.time())
        client.record_response(error.code, "pdf_http_error", started, delay, error.headers)
        if error.code == 429:
            client.gate.slow_down()
        if error.code in {408, 429, 500, 502, 503, 504}:
            client.gate.defer(max(60, delay or 0))
        error.close()
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def recover(store, client, directory, *, limit=5, max_pages=50, ocr=True, max_new_ocr_pages=20,
            cancelled=None, opener=None, progress=None, max_bytes=MAX_BYTES, order="oldest"):
    validate_pdf_bytes(max_bytes)
    if type(limit) is not int or not 1 <= limit <= 100 or type(max_pages) is not int or not 1 <= max_pages <= 5000:
        raise ValueError("Use 1–100 documents and 1–5,000 pages per PDF")
    if type(max_new_ocr_pages) is not int or not 1 <= max_new_ocr_pages <= 500:
        raise ValueError("Use 1–500 new OCR pages per document/run")
    if order not in ("oldest", "smallest-first"):
        raise ValueError("Use oldest or smallest-first recovery order")
    directory = Path(directory)
    require_space(store.path.parent)
    require_space(directory if directory.exists() else directory.parent)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another PDF recovery worker is running") from None
        if progress:
            progress({"state": "starting", "selection_order": order, "full_archive_verified": False})
        result = _recover(store, client, directory, limit, max_pages, ocr, max_new_ocr_pages, cancelled, opener, progress, max_bytes, order)
        if progress:
            progress(result)
        return result


def reported_size_priority(row):
    """Catalog bytes are a scheduling hint, never permission to trust a download."""
    try:
        metadata = json.loads(row["metadata"])
        size, state = catalog_number(metadata.get("DocumentSize"))
    except (TypeError, ValueError, AttributeError):
        size, state = None, "invalid"
    return (0, size, row["queue_row"]) if state == "positive" else (1, 0, row["queue_row"])


def _recover(store, client, directory, limit, max_pages, ocr, max_new_ocr_pages, cancelled, opener, progress, max_bytes, order):
    store.db.executescript("""CREATE TABLE IF NOT EXISTS pdf_recovery(
      accession TEXT PRIMARY KEY, outcome TEXT NOT NULL, event_count INTEGER NOT NULL DEFAULT 0,
      next_retry REAL, source_url TEXT, sha256 TEXT, bytes INTEGER, detail TEXT,
      policy TEXT NOT NULL,
      updated TEXT DEFAULT CURRENT_TIMESTAMP);
      CREATE TABLE IF NOT EXISTS pdf_recovery_events(
        id INTEGER PRIMARY KEY, accession TEXT NOT NULL, outcome TEXT NOT NULL,
        source_url TEXT, sha256 TEXT, detail TEXT, captured TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    result = {"state": "batch_limit", "attempted": 0, "recovered": 0, "outcomes": [],
              "selection_order": order, "full_archive_verified": False}
    policy = json.dumps({"version": 4, "max_pages": max_pages, "ocr": ocr, "max_bytes": max_bytes,
                         "max_new_ocr_pages":max_new_ocr_pages}, sort_keys=True)
    query = RECOVERY_CANDIDATES_SQL
    parameters = (policy, time.time(), policy)
    if order == "oldest":
        rows = store.db.execute(query + " ORDER BY queue_row LIMIT ?", (*parameters, limit)).fetchall()
    else:
        cursor = store.db.execute(query, parameters)
        def candidates():
            for row in cursor:
                if cancelled and cancelled():
                    raise InterruptedError("PDF recovery stopped")
                yield row
        try:
            # Stream eligible gaps and retain at most `limit` rows in memory.
            # Larger and unknown-size files remain eligible for later batches.
            rows = heapq.nsmallest(limit, candidates(), key=reported_size_priority)
        except InterruptedError:
            result["state"] = "stopped"
            return result
        finally:
            cursor.close()
    def record(accession, outcome, url=None, retained=None, detail=None, retry=None):
        retained = retained or {}
        with store.db:
            store.db.execute("""INSERT INTO pdf_recovery(accession,outcome,event_count,next_retry,source_url,sha256,bytes,detail,policy)
                VALUES(?,?,1,?,?,?,?,?,?) ON CONFLICT(accession) DO UPDATE SET outcome=excluded.outcome,
                event_count=event_count+1,next_retry=excluded.next_retry,source_url=coalesce(excluded.source_url,source_url),
                sha256=coalesce(excluded.sha256,sha256),bytes=coalesce(excluded.bytes,bytes),detail=excluded.detail,
                policy=excluded.policy,updated=CURRENT_TIMESTAMP""",
                (accession, outcome, retry, url, retained.get("sha256"), retained.get("bytes"), detail, policy))
            store.db.execute("INSERT INTO pdf_recovery_events(accession,outcome,source_url,sha256,detail) VALUES(?,?,?,?,?)",
                             (accession, outcome, url, retained.get("sha256"), detail))
        result["outcomes"].append({"accession": accession, "outcome": outcome, "detail": detail})
    for row in rows:
        if cancelled and cancelled():
            result["state"] = "stopped"
            break
        accession, url, retained = row["accession"], None, None
        last_progress = {}
        def report_page(detail):
            nonlocal last_progress
            last_progress = dict(detail)
            if progress:
                progress({**result, "state": "running", "current_document": detail})
        try:
            require_space(store.path.parent)
            require_space(directory)
            metadata = json.loads(row["metadata"])
            url = official_url(accession, metadata)
            # Catalog page counts can differ from the actual PDF. The bounded
            # download and parser enforce limits against the retained source.
            result["attempted"] += 1
            report_page({"accession": accession, "phase": "retaining_pdf", "partial_text_published": False})
            target = directory / (accession + ".pdf")
            if target.exists():
                if target.stat().st_size > max_bytes:
                    raise RecoveryReview("Retained PDF exceeds the file allowance")
                candidate = {"sha256": file_sha256(target,max_bytes), "bytes": target.stat().st_size}
                if row["sha256"] and candidate["sha256"] != row["sha256"]:
                    raise RecoveryReview("Retained PDF checksum changed; original audit is preserved")
                retained = candidate
            else:
                retained = download_pdf(url, target, client, opener=opener, cancelled=cancelled,
                                        max_bytes=max_bytes,
                                        progress=lambda detail: report_page({"accession": accession, **detail}))
            # Retaining a PDF is a separate checkpoint from recovering its text.
            record(accession, "pdf_retained", url, retained, retry=time.time())
            imported = import_pdf(store, target, accession, ocr=ocr, max_pages=max_pages,
                                  max_new_ocr_pages=max_new_ocr_pages, cancelled=cancelled,
                                  progress=report_page, max_bytes=max_bytes)
            current = store.db.execute("SELECT status FROM documents WHERE accession=?", (accession,)).fetchone()[0]
            outcome = ("indexed_with_gaps" if imported["blank_pages_needing_review"] else "indexed_unverified") if current == "indexed" else "no_readable_text"
            record(accession, outcome, url, retained,
                   json.dumps({"pages": imported["pages"], "ocr_pages": imported["ocr_pages"],
                               "blank_pages": imported["blank_pages_needing_review"],
                               "source_metadata_rechecks": imported.get("source_metadata_rechecks",0)}))
            result["recovered"] += int(current == "indexed")
        except PDFExtractionPending as error:
            record(accession,"ocr_pending",url,retained,json.dumps(error.progress),retry=time.time())
        except StoragePause as error:
            result.update(state="storage_pause", detail=str(error))
            break
        except (BudgetStop, InterruptedError) as error:
            result.update(state="stopped" if isinstance(error, InterruptedError) else "budget_pause", detail=str(error))
            break
        except urllib.error.HTTPError as error:
            code = error.code
            record(accession, "http_error", url, retained, f"HTTP {code}", time.time() + (86400 if code in {403,404,410} else 300))
            if code not in {404, 410}:
                result.update(state="network_wait" if code in {408,429,500,502,503,504} else "needs_review", detail=f"HTTP {code}; this recovery run stopped")
                break
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            record(accession, "network_or_io_error", url, retained, "Request or file operation failed", time.time() + 300)
            result["state"] = "network_wait"
            break
        except (RecoveryReview, PDFImportLimit) as error:
            record(accession, "needs_review", url, retained, str(error))
        except (RuntimeError, ImportError) as error:
            # Dependencies or a busy OCR worker may recover; preserve the PDF and pause.
            record(accession, "processing_deferred", url, retained, extraction_failure(error,last_progress), time.time() + 300)
            result["state"] = "needs_review"
            break
        except Exception as error:
            # Parser failures and unexpected failures retain a reviewable outcome.
            record(accession, "needs_review", url, retained, extraction_failure(error,last_progress))
            result["state"] = "needs_review"
            break
    else:
        if any(r["outcome"] == "ocr_pending" for r in result["outcomes"]):
            result["state"] = "ocr_pending"
        elif len(rows) < limit:
            result["state"] = "eligible_queue_exhausted"
    return result


def main(argv=None):
    from .client import AdamsClient
    from .config import load_api_key
    from .store import Store
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research.sqlite")
    parser.add_argument("--rate-db", default="data/adams.sqlite")
    parser.add_argument("--directory", default="data/recovery")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--max-mib", type=int, default=25, help="Per-PDF file allowance (1–512 MiB)")
    parser.add_argument("--max-new-ocr-pages", type=int, default=20,
                        help="New OCR pages per document/run; completed page caches do not consume this budget")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--order", choices=("oldest", "smallest-first"), default="oldest",
                        help="Prioritize catalog-reported small PDFs without removing larger or unknown-size gaps")
    parser.add_argument("--rps", type=float, default=2, help="Local ceiling; shared NRC cooldowns still apply")
    parser.add_argument("--daily-limit", type=int, default=90000, help="Shared local rolling 24-hour request budget")
    args = parser.parse_args(argv)
    if not 0 < args.rps <= 6 or args.daily_limit < 1:
        parser.error("Use 0 < rps <= 6 and a positive daily request budget")
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    directory = Path(args.directory)
    def cancelled():
        return stopping[0] or (directory / "STOP").exists()
    def report(value):
        value = {**value, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "pid": os.getpid(), "free_bytes": shutil.disk_usage(directory).free,
                 "reserve_bytes": int(reserve_gb() * 1e9)}
        temporary = directory / "STATUS.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(directory / "STATUS.json")
    store, rate = Store(args.db), Store(args.rate_db)
    try:
        client = AdamsClient(rate, load_api_key(), max_requests=args.limit, rps=args.rps,
                             daily_limit=args.daily_limit, cancelled=cancelled)
        try:
            result = recover(store, client, directory, limit=args.limit, max_pages=args.max_pages,
                             max_new_ocr_pages=args.max_new_ocr_pages, ocr=not args.no_ocr,
                             cancelled=cancelled, progress=report, max_bytes=args.max_mib * 1024 * 1024,
                             order=args.order)
        except StoragePause as error:
            result = {"state": "storage_pause", "detail": str(error), "full_archive_verified": False}
            if directory.exists():
                report(result)
        result.update(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(),
                      free_bytes=shutil.disk_usage(store.path.parent).free)
        print(json.dumps(result, indent=2))
    finally:
        store.db.close()
        rate.db.close()


if __name__ == "__main__":
    main()
