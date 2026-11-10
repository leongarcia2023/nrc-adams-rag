from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import tempfile
from pathlib import Path

from .client import unpack_search
from .store import digest
from .pdf_policy import DEFAULT_PDF_BYTES


class PDFImportLimit(ValueError):
    """A bounded PDF import needs a different processing allowance."""


def embedded_text_recovery(text):
    """Recognize a strong encoding failure without guessing the intended text.

    Ordinary layout whitespace and sparse stray controls are not sufficient.
    This catches dense control-code extraction, not every unreadable font map.
    """
    controls = sum((ord(c) < 32 or 127 <= ord(c) < 160)
                   and c not in "\t\n\r\f\v\x85" for c in text)
    if controls >= 5 and controls * 100 >= len(text):
        return {"reason": "dense_nonprinting_controls", "characters": len(text),
                "control_characters": controls, "text_sha256": digest(text)}
    return None


class PDFExtractionPending(RuntimeError):
    """Completed OCR pages are retained; no partial document was published."""
    def __init__(self, accession, pages, next_page, new_ocr_pages):
        self.progress = {"accession":accession,"pages":pages,"next_page":next_page,
                         "new_ocr_pages":new_ocr_pages,"partial_text_published":False}
        super().__init__(f"OCR page budget reached before page {next_page} of {pages}; rerun to resume cached pages")


def archive(store, payload):
    """Content-addressed raw responses permit later reprocessing without API calls."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    root = store.path.parent / "raw"
    checksum = digest(raw)
    # Preserve existing flat archives; shard new ones to avoid millions of
    # entries in one filesystem directory at full ADAMS scale.
    legacy = root / (checksum + ".json.gz")
    if legacy.exists():
        return
    target = root / checksum[:2] / checksum[2:4] / (checksum + ".json.gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        fd, name = tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(gzip.compress(raw.encode()))
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def crawl(store, client, start, end, library="main", docket=None, max_offset=1000, refresh=False):
    """Daily partitions with atomic checkpoints; exhaustion is not proof of completeness.

    Conservative offset ceiling flags dense days for narrower follow-up queries.
    A production harvester must reconcile these with an authoritative manifest.
    """
    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if first > last or max_offset < 1:
        raise ValueError("Invalid date interval or offset ceiling")
    job = json.dumps({"library": library, "docket": docket, "date_filter_version": 2}, sort_keys=True)
    if refresh:
        with store.db:
            store.db.execute("DELETE FROM crawl WHERE job=? AND day BETWEEN ? AND ?", (job, start, end))
    day = first
    while day <= last:
        date = day.isoformat()
        with store.db:
            store.db.execute("INSERT OR IGNORE INTO crawl(job,day) VALUES(?,?)", (job, date))
        row = store.db.execute("SELECT * FROM crawl WHERE job=? AND day=?", (job, date)).fetchone()
        if row["state"] in {"exhausted_unverified", "blocked"}:
            day += dt.timedelta(days=1)
            continue
        offset, last_hash = row["offset"], row["last_page_hash"]
        while True:
            if offset >= max_offset:
                with store.db:
                    store.db.execute("UPDATE crawl SET state='blocked',note=? WHERE job=? AND day=?",
                                     ("Offset ceiling reached; narrow by docket or raise only after verifying APS limits", job, date))
                break
            # APS compares timestamps, so equality to a date misses daytime additions.
            # The documented ge/le operators give a full day plus the next midnight;
            # boundary overlap is harmless because accession imports are idempotent.
            tomorrow = (day + dt.timedelta(days=1)).isoformat()
            filters = [{"field": "DateAddedTimestamp", "value":
                        f"(DateAddedTimestamp ge '{date}') and (DateAddedTimestamp le '{tomorrow}')"}]
            if docket:
                filters.append({"field": "DocketNumber", "value": docket, "operator": "contains"})
            response = client.search(filters=filters, skip=offset, library=library)
            docs = unpack_search(response)
            archive(store, response)
            if not docs:
                # Counts may be capped or absent. No empty response is labeled 'complete'.
                count = response.get("count")
                state = "blocked" if isinstance(count, int) and count > offset else "exhausted_unverified"
                with store.db:
                    store.db.execute("UPDATE crawl SET state=?,note=? WHERE job=? AND day=?",
                                     (state, "Empty page observed; reconcile separately for coverage", job, date))
                break
            ids = [doc["AccessionNumber"] for doc in docs]
            page_hash = digest(json.dumps(ids))
            if page_hash == last_hash or len(ids) != len(set(ids)):
                with store.db:
                    store.db.execute("UPDATE crawl SET state='blocked',note=? WHERE job=? AND day=?",
                                     ("Repeated page or duplicate IDs; pagination may be truncated/unstable", job, date))
                break
            with store.db:
                for doc in docs:
                    store.put(doc, "api_search_excerpt")
                offset += len(docs)
                store.db.execute("UPDATE crawl SET offset=?,state='running',last_page_hash=? WHERE job=? AND day=?",
                                 (offset, page_hash, job, date))
            last_hash = page_hash
        day += dt.timedelta(days=1)
    return store.status()


def live_search(store, client, query, library="both", docket=None):
    filters = [{"field": "DocketNumber", "value": docket, "operator": "contains"}] if docket else []
    response = client.search(q=query, filters=filters, library=library)
    docs = unpack_search(response)
    archive(store, response)
    with store.db:
        for doc in docs:
            store.put(doc, "api_search_excerpt")
    return len(docs)


def hydrate(store, client, limit=20, retry_missing=False):
    if limit < 1:
        raise ValueError("Hydration limit must be positive")
    states = ("missing_text", "search_excerpt", "unavailable_text") if retry_missing else ("missing_text", "search_excerpt")
    rows = store.db.execute("SELECT accession FROM documents WHERE status IN (" +
                            ",".join("?" for _ in states) + ") LIMIT ?", (*states, limit)).fetchall()
    for row in rows:
        response = client.request(accession=row["accession"])
        if not isinstance(response, dict) or not isinstance(response.get("document"), dict):
            raise ValueError("Unexpected APS Get Document response")
        doc = response["document"]
        if doc.get("AccessionNumber") != row["accession"]:
            raise ValueError("APS returned a different accession than requested")
        archive(store, response)
        with store.db:
            status = store.put(doc, "api_document_unverified")
            if status == "missing_text":
                store.db.execute("UPDATE documents SET status='unavailable_text' WHERE accession=?", (row["accession"],))
            store.db.execute("INSERT INTO retrieval_events(accession,outcome) VALUES(?,?)", (row["accession"], status))
    return store.status()


def import_jsonl(store, path):
    count = 0
    with open(path, encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                archive(store, doc)
                with store.db:
                    store.put(doc, "jsonl_import")
                count += 1
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid JSONL record at line {number}: {error}") from error
    return count


def import_pdf(store, path, accession, title=None, url=None, docket=None, *, ocr=False, max_pages=50,
               max_new_ocr_pages=20, cancelled=None, progress=None, max_bytes=DEFAULT_PDF_BYTES):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("PDF import needs pypdf; install with: python3 -m pip install '.[pdf]'") from None
    from .storage import require_space
    from .pdf_policy import validate_pdf_bytes, file_sha256
    validate_pdf_bytes(max_bytes)
    path = Path(path).resolve(strict=True)
    if type(max_pages) is not int or not 1 <= max_pages <= 5000:
        raise ValueError("PDF page allowance must be between 1 and 5,000")
    if type(max_new_ocr_pages) is not int or not 1 <= max_new_ocr_pages <= 500:
        raise ValueError("Use a new OCR page budget between 1 and 500 per document/run")
    if path.stat().st_size > max_bytes:
        raise PDFImportLimit(f"PDF exceeds the {max_bytes / 1024**2:g} MiB import allowance")
    require_space(store.path.parent, extra_bytes=max(32 * 1024 * 1024, 2 * path.stat().st_size))
    from .ocr import PDFSourceSnapshot
    snapshot = PDFSourceSnapshot(path, max_bytes=max_bytes)
    checksum = snapshot.sha256
    reader = PdfReader(path)
    if not 1 <= len(reader.pages) <= max_pages:
        raise PDFImportLimit(f"PDF has {len(reader.pages)} pages; this import permits 1–{max_pages}")
    pages, extraction, recognized = [], [], []
    new_ocr_pages = [0]
    def report(phase):
        if progress:
            progress({"accession": accession, "phase": phase, "pages": len(reader.pages),
                      "completed_pages": len(pages), "new_ocr_pages": new_ocr_pages[0],
                      "cached_ocr_pages": len(recognized) - new_ocr_pages[0],
                      "partial_text_published": False})
    report("extracting")
    def before_recognize(number):
        if new_ocr_pages[0] >= max_new_ocr_pages:
            raise PDFExtractionPending(accession,len(reader.pages),number,new_ocr_pages[0])
        new_ocr_pages[0] += 1
    for number, page in enumerate(reader.pages, 1):
        require_space(store.path.parent)
        snapshot.check()
        if cancelled and cancelled():
            raise InterruptedError("PDF import cancelled; existing evidence is unchanged")
        text = page.extract_text() or ""
        method = {"page": number, "method": "embedded_text"}
        recovery = embedded_text_recovery(text)
        if recovery and not ocr:
            raise PDFImportLimit(f"PDF page {number} has unreadable embedded character encoding; enable OCR or supply a reviewed page transcription")
        if ocr and (not text.strip() or recovery):
            from .ocr import cached_page
            result = cached_page(path, number, store.path.parent, cancelled=cancelled,
                                 before_recognize=before_recognize,snapshot=snapshot)
            text = result["text"]
            if embedded_text_recovery(text):
                raise ValueError(f"OCR page {number} still contains unreadable character encoding; source review is required")
            recognized.append(result)
            method = {key: value for key, value in result.items() if key not in {"text", "lines"}}
            method["recognized_lines"] = len(result["lines"])
            if recovery:
                method["embedded_text_recovery"] = recovery
        pages.append(text)
        extraction.append(method)
        report("extracting")
    snapshot.check()
    if checksum != file_sha256(path, max_bytes):
        raise ValueError("PDF changed during extraction; no evidence was imported")
    # Merge at publication time, so OCR cannot erase newly refreshed catalog fields.
    if cancelled and cancelled():
        raise InterruptedError("PDF import cancelled; existing evidence is unchanged")
    require_space(store.path.parent, extra_bytes=max(32 * 1024 * 1024, sum(len(page.encode()) for page in pages) * 12))
    report("publishing")
    with store.db:
        store.db.execute("BEGIN IMMEDIATE")
        old = store.db.execute("SELECT metadata,title,url,text_hash FROM documents WHERE accession=?", (accession,)).fetchone()
        doc = json.loads(old["metadata"]) if old else {}
        from .extraction_correction import has_reviewed_version
        reviewed = old and has_reviewed_version(store.db, accession, old["text_hash"])
        if reviewed and (doc.get("LocalFileSHA256") != checksum or old["text_hash"] != digest("\f".join(pages))):
            raise ValueError("Reviewed page corrections require a new review before replacement by automatic PDF extraction")
        from .extraction_review import for_version
        observed = old and for_version(store.db, accession, old["text_hash"])
        if observed and (doc.get("LocalFileSHA256") != checksum or old["text_hash"] != digest("\f".join(pages))):
            raise ValueError("Existing extraction reviews require a new review before replacement by automatic PDF extraction")
        reviewed_methods = doc.get("PageExtraction") if reviewed else None
        doc.update(AccessionNumber=accession, LocalFileSHA256=checksum, ExtractedPageCount=len(pages),
                   BlankExtractedPages=[i for i, text in enumerate(pages, 1) if not text.strip()],
                   PageExtraction=extraction, SourceFileMetadataRechecks=snapshot.metadata_rechecks)
        if reviewed_methods is not None:
            doc["PageExtraction"] = reviewed_methods
        doc["DocumentTitle"] = title if title is not None else (old["title"] if old else accession)
        doc["Url"] = url if url is not None else (old["url"] if old else None)
        if docket is not None:
            doc["DocketNumber"] = docket
        doc.setdefault("EstimatedPageCount", len(pages))
        archive(store, {**doc, "pages": pages, "ocr_results": recognized})
        require_space(store.path.parent)
        store.put(doc, "pdf", pages)
        store.db.execute("INSERT INTO retrieval_events(accession,outcome) VALUES(?,?)", (accession, "pdf_ocr_import" if recognized else "pdf_import"))
    return {"accession": accession, "pages": len(pages), "ocr_pages": len(recognized),
            "new_ocr_pages": new_ocr_pages[0],
            "source_metadata_rechecks": snapshot.metadata_rechecks,
            "blank_pages_needing_review": doc["BlankExtractedPages"], "extraction_verified": False}
