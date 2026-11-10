"""Publish reviewed PDF page transcriptions without erasing prior extraction history."""
from .source_content import read_text
import argparse
import datetime as dt
import json
from pathlib import Path
import re

from .extraction_review import REASONS, for_version
from .ingest import archive
from .ocr import PDFSourceSnapshot
from .pdf_policy import DEFAULT_PDF_BYTES, file_sha256
from .storage import require_space
from .store import Store, digest
from .page_boundaries import retained_pdf_pages


def has_reviewed_version(db, accession, text_hash):
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_corrections'").fetchone()
    return bool(exists and db.execute(
        "SELECT 1 FROM extraction_corrections WHERE accession=? AND new_hash=? LIMIT 1",
        (accession, text_hash)).fetchone())


def validate_manifest(value):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Use a version 1 reviewed-page correction manifest")
    for key in ("accession", "reviewer", "reviewed_utc"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError("Correction identity and reviewer fields are required")
    if dt.datetime.fromisoformat(value["reviewed_utc"]).tzinfo is None:
        raise ValueError("The review timestamp requires a timezone")
    for key in ("base_text_hash", "pdf_sha256"):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError("Corrections must identify exact source and extraction hashes")
    pages = value.get("pages")
    if not isinstance(pages, list) or not 1 <= len(pages) <= 100:
        raise ValueError("Review between one and 100 complete physical pages per correction")
    seen = set()
    for page in pages:
        if not isinstance(page, dict) or type(page.get("page")) is not int or not 1 <= page["page"] <= 5000 or page["page"] in seen:
            raise ValueError("Correction pages must be distinct physical page numbers")
        seen.add(page["page"])
        if (not isinstance(page.get("text"), str) or not page["text"].strip()
                or "\f" in page["text"] or len(page["text"].encode()) > 1024 * 1024):
            raise ValueError("Use a nonempty complete page transcription within 1 MiB, without page separators")
        if not isinstance(page.get("before_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", page["before_sha256"]):
            raise ValueError("Each correction must identify the exact old page text")
        reasons = page.get("resolved_reasons")
        if not isinstance(reasons, list) or any(not isinstance(r, str) or r not in REASONS for r in reasons):
            raise ValueError("Use known extraction-review reasons")
        if not isinstance(page.get("review_note"), str) or not 1 <= len(page["review_note"].strip()) <= 2000:
            raise ValueError("Each corrected page needs a review observation")
    if len(json.dumps(value).encode()) > 8 * 1024 * 1024:
        raise ValueError("Correction manifest exceeds 8 MiB")


def apply_correction(store, pdf, manifest, max_bytes=DEFAULT_PDF_BYTES):
    validate_manifest(manifest)
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    correction_id = digest(canonical)
    require_space(store.path.parent, extra_bytes=max(32 * 1024 * 1024, len(canonical.encode()) * 12))
    source = PDFSourceSnapshot(pdf, max_bytes=max_bytes)
    if source.sha256 != manifest["pdf_sha256"]:
        raise ValueError("The retained PDF does not match the reviewed source")
    accession = manifest["accession"]
    with store.db:
        store.db.execute("BEGIN IMMEDIATE")
        store.db.execute("""CREATE TABLE IF NOT EXISTS extraction_corrections(
            id TEXT PRIMARY KEY,accession TEXT NOT NULL,old_hash TEXT NOT NULL,new_hash TEXT NOT NULL,
            pdf_sha256 TEXT NOT NULL,manifest TEXT NOT NULL,created TEXT DEFAULT CURRENT_TIMESTAMP)""")
        store.db.execute("CREATE INDEX IF NOT EXISTS corrections_by_version ON extraction_corrections(accession,new_hash)")
        old = store.db.execute("SELECT * FROM documents WHERE accession=?", (accession,)).fetchone()
        if not old or old["status"] != "indexed" or old["provenance"] != "pdf":
            raise ValueError("A retained, indexed PDF extraction is required")
        applied = store.db.execute("SELECT new_hash FROM extraction_corrections WHERE id=?", (correction_id,)).fetchone()
        if applied:
            if old["text_hash"] != applied[0]:
                raise ValueError("This correction was applied, but the document has since changed")
            return {"state": "already_applied", "accession": accession, "correction_id": correction_id, "text_hash": applied[0]}
        if old["text_hash"] != manifest["base_text_hash"]:
            raise ValueError("The extraction changed; review the current version before applying corrections")
        metadata = json.loads(old["metadata"])
        if metadata.get("LocalFileSHA256") != source.sha256:
            raise ValueError("The PDF identity differs from the retained extraction")
        version = store.db.execute("SELECT content,metadata FROM document_versions WHERE accession=? AND text_hash=? ORDER BY captured DESC,rowid DESC LIMIT 1",
                                   (accession, old["text_hash"])).fetchone()
        if not version:
            raise ValueError("The retained source version is missing")
        body = read_text(version[0], store.db)
        if digest(body) != old["text_hash"]:
            raise ValueError("The retained source version failed its text checksum")
        pages = retained_pdf_pages(body, json.loads(version[1]))
        changes = {page["page"]: page for page in manifest["pages"]}
        for number, page in changes.items():
            if number > len(pages) or digest(pages[number - 1]) != page["before_sha256"]:
                raise ValueError("A physical page differs from the reviewed extraction")
            pages[number - 1] = page["text"]
        new_hash = digest("\f".join(pages))
        if new_hash == old["text_hash"]:
            raise ValueError("The correction does not change the retained page text")
        reviews = for_version(store.db, accession, old["text_hash"])
        methods = {p["page"]: p for p in metadata.get("PageExtraction", [])}
        for number, page in changes.items():
            methods[number] = {"page": number, "method": "reviewed_transcription",
                "reviewer": manifest["reviewer"], "reviewed_utc": manifest["reviewed_utc"],
                "review_note": page["review_note"], "correction_id": correction_id,
                "source_pdf_sha256": source.sha256, "previous_page_sha256": page["before_sha256"]}
        metadata.update(PageExtraction=[methods[n] for n in sorted(methods)],
            ExtractedPageCount=len(pages), BlankExtractedPages=[i for i, p in enumerate(pages, 1) if not p.strip()],
            ReviewedPageCorrectionIds=list(dict.fromkeys(metadata.get("ReviewedPageCorrectionIds", []) + [correction_id])))
        require_space(store.path.parent, extra_bytes=max(32 * 1024 * 1024, sum(len(p.encode()) for p in pages) * 12))
        source.check()
        if file_sha256(source.path, max_bytes) != source.sha256:
            raise ValueError("The source PDF changed during correction")
        archive(store, {"kind": "reviewed_page_correction", "manifest": manifest, "pages": pages})
        require_space(store.path.parent)
        store.put(metadata, "pdf", pages)
        carried = []
        for review in reviews:
            page = changes.get(review["page"])
            remaining = set(review["reasons"]) - (set(page["resolved_reasons"]) if page else set())
            if not remaining:
                continue
            # Unchanged pages and whole-document observations remain in force.
            # A changed page clears only issues explicitly resolved by its review.
            store.db.execute("INSERT INTO extraction_reviews(accession,text_hash,page,disposition,reasons,note) VALUES(?,?,?,?,?,?)",
                (accession, new_hash, review["page"], review["disposition"], json.dumps(sorted(remaining)), review["note"]))
            carried.append(review["page"])
        store.db.execute("INSERT INTO extraction_corrections(id,accession,old_hash,new_hash,pdf_sha256,manifest) VALUES(?,?,?,?,?,?)",
            (correction_id, accession, old["text_hash"], new_hash, source.sha256, canonical))
        store.db.execute("INSERT INTO retrieval_events(accession,outcome) VALUES(?,?)", (accession, "pdf_reviewed_correction"))
    return {"state": "applied", "accession": accession, "correction_id": correction_id,
            "old_text_hash": old["text_hash"], "text_hash": new_hash,
            "corrected_pages": sorted(changes), "carried_review_pages": carried, "document_extraction_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research.sqlite")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-mib", type=int, default=25)
    args = parser.parse_args(argv)
    path = Path(args.manifest)
    if path.stat().st_size > 8 * 1024 * 1024:
        parser.error("Correction manifest exceeds 8 MiB")
    store = Store(args.db)
    try:
        print(json.dumps(apply_correction(store, args.pdf, json.loads(path.read_text()), args.max_mib * 1024 * 1024), indent=2))
    finally:
        store.db.close()


if __name__ == "__main__":
    main()
