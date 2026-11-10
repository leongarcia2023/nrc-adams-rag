"""Version-bound extraction diagnostics; presence of text is never proof of completeness."""
from .source_content import read_text
import argparse
import hashlib
import json
import re

POLICY_VERSION = 2
UNAVAILABLE_NOTICE = (
    "The contents for this document could not be indexed due to compatibility and/or size restrictions. "
    "We have however indexed the metadata properties to make the document findable. "
    "To view the contents please click the document title link which will download or open the document per your browser settings"
)


def is_unavailable_notice(text):
    """Match the observed whole-body service notice, never a report quoting it."""
    return " ".join(text.split()).casefold().rstrip(".") == UNAVAILABLE_NOTICE.casefold()


MESSAGES = {
    "reviewed_extraction_issue": "A review found extraction errors; consult the page-specific observations and original PDF.",
    "no_readable_text": "No readable text was retained.",
    "blank_pdf_pages": "Some PDF pages yielded no text; they may be scans or intentionally blank pages.",
    "sparse_text_for_page_count": "Very little text was retained relative to the reported page count; review the original.",
    "replacement_characters": "The extraction contains replacement characters that may hide source content.",
    "page_count_mismatch": "The extracted PDF page count differs from the catalog's reported count.",
    "retained_text_unavailable": "The retained source text is unavailable for assessment.",
    "machine_ocr": "Some pages use machine-recognized text; verify numbers, identifiers and layout against the original PDF.",
    "source_unavailable_notice": "The source returned an indexing-unavailable notice instead of document text.",
    "legacy_content_scope_unverified": "Legacy indexed text may be an abstract or a full document; its completeness has not been verified.",
}


def metadata_hash(metadata):
    return hashlib.sha256(metadata.encode()).hexdigest()


def reported_pages(metadata):
    value = metadata.get("EstimatedPageCount")
    if isinstance(value, bool):
        return None
    try:
        count = int(str(value).strip().replace(",", ""))
        return count if 1 <= count <= 1_000_000 else None
    except (ValueError, TypeError):
        return None


def classify(stats, metadata, provenance, status):
    result = {**stats, "policy_version": POLICY_VERSION, "reported_pages": reported_pages(metadata),
              "scope": "section" if metadata.get("section_id") is not None else "document",
              "source_complete_verified": False, "extraction_verified": False, "flags": []}
    if status in {"package", "search_excerpt", "legacy_metadata"} or provenance.startswith("api_search"):
        result.update(state="not_evidence", scope="metadata")
        return result
    if result.get("service_notice") or (metadata.get("SourceContentAvailabilityNotice") and not result["nonspace_characters"]):
        result.update(state="no_text", flags=["source_unavailable_notice"])
        return result
    if not result.get("text_available", True):
        result.update(state="not_assessed", flags=["retained_text_unavailable"])
        return result
    if not result["nonspace_characters"]:
        result.update(state="no_text", flags=["no_readable_text"])
        return result
    flags = result["flags"]
    if metadata.get("SourceContentScope") == "legacy_indexed_text_unverified":
        result["scope"] = "legacy_indexed_text"
        flags.append("legacy_content_scope_unverified")
    extraction = metadata.get("PageExtraction")
    ocr_pages = [item["page"] for item in (extraction if isinstance(extraction, list) else [])
                 if isinstance(item, dict) and item.get("method") == "apple_vision_ocr" and type(item.get("page")) is int]
    result.update(ocr_page_count=len(ocr_pages), ocr_pages=ocr_pages[:100])
    if ocr_pages:
        flags.append("machine_ocr")
    if result.get("blank_page_count", 0):
        flags.append("blank_pdf_pages")
    pages = result["reported_pages"]
    if result["scope"] in {"document", "legacy_indexed_text"} and pages and pages >= 5 and result["words"] < pages * 30:
        flags.append("sparse_text_for_page_count")
    if result["replacement_characters"] >= 5:
        flags.append("replacement_characters")
    if pages and result.get("extracted_pages") is not None and pages != result["extracted_pages"]:
        flags.append("page_count_mismatch")
    result["state"] = "needs_review" if flags else "text_present_unverified"
    return result


def assess(text, metadata, provenance, status, pages=None, text_available=True):
    blank = [i for i, page in enumerate(pages, 1) if not page.strip()] if pages is not None else []
    stats = {"characters": len(text), "nonspace_characters": sum(not c.isspace() for c in text),
             "service_notice": is_unavailable_notice(text),
             "words": sum(1 for _ in re.finditer(r"\S+", text)),
             "replacement_characters": text.count("\ufffd"), "text_available": text_available,
             "extracted_pages": len(pages) if pages is not None else None,
             "blank_page_count": len(blank), "blank_pages": blank[:100],
             "blank_page_list_truncated": len(blank) > 100, "page_mapping_available": pages is not None}
    return classify(stats, metadata, provenance, status)


def save(db, accession, text_hash, metadata, report, provenance):
    """The conditional write also protects a migration from concurrent source replacement."""
    cursor = db.execute("""INSERT INTO document_quality(accession,text_hash,metadata_hash,policy_version,state,report)
        SELECT accession,text_hash,?,?,?,? FROM documents WHERE accession=? AND text_hash=? AND metadata=? AND provenance=?
        ON CONFLICT(accession) DO UPDATE SET text_hash=excluded.text_hash,metadata_hash=excluded.metadata_hash,
        policy_version=excluded.policy_version,state=excluded.state,report=excluded.report""",
        (metadata_hash(metadata), POLICY_VERSION, report["state"], json.dumps(report), accession, text_hash, metadata, provenance))
    return cursor.rowcount


def get(db, accession):
    row = db.execute("""SELECT q.*,d.text_hash AS current_hash,d.metadata FROM documents d
                        LEFT JOIN document_quality q USING(accession) WHERE d.accession=?""", (accession,)).fetchone()
    result = {"state": "not_assessed", "flags": [], "source_complete_verified": False, "extraction_verified": False}
    if (row and row["policy_version"] == POLICY_VERSION and row["text_hash"] == row["current_hash"]
            and row["metadata_hash"] == metadata_hash(row["metadata"])):
        result = json.loads(row["report"])
    if row:
        from .extraction_review import annotate
        result = annotate(result, db, accession, row["current_hash"])
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='markdown_checks'").fetchone():
            check = db.execute('SELECT report FROM markdown_checks WHERE accession=? AND text_hash=?', (accession, row['current_hash'])).fetchone()
            if check:
                result['document_check'] = json.loads(check['report'])
    return result


def migrate(store, limit=1000, cancelled=None):
    from .storage import StoragePause, require_space
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("Use an assessment limit between 1 and 10,000")
    result = {"assessed": 0, "stale_skipped": 0, "state": "batch_limit", "full_archive_verified": False}
    try:
        require_space(store.path.parent)
    except StoragePause:
        result["state"] = "storage_pause"
        return result
    rows = store.db.execute("""SELECT d.* FROM documents d LEFT JOIN document_quality q USING(accession)
        WHERE q.accession IS NULL OR q.policy_version<>? OR q.text_hash<>d.text_hash
        ORDER BY d.rowid LIMIT ?""", (POLICY_VERSION, limit)).fetchall()
    for row in rows:
        if cancelled and cancelled():
            result["state"] = "stopped"
            break
        try:
            require_space(store.path.parent)
        except StoragePause:
            result["state"] = "storage_pause"
            break
        metadata = json.loads(row["metadata"])
        version = store.db.execute("SELECT content,metadata FROM document_versions WHERE accession=? AND text_hash=? ORDER BY captured DESC,rowid DESC LIMIT 1",
                                   (row["accession"], row["text_hash"])).fetchone()
        text = read_text(version[0], store.db) if version else ""
        from .page_boundaries import retained_pdf_pages
        pages = retained_pdf_pages(text, json.loads(version[1])) if version and row["provenance"] == "pdf" else None
        report = assess(text, metadata, row["provenance"], row["status"], pages,
                        text_available=bool(version) or row["status"] != "indexed")
        try:
            require_space(store.path.parent)
        except StoragePause:
            result["state"] = "storage_pause"
            break
        with store.db:
            changed = save(store.db, row["accession"], row["text_hash"], row["metadata"], report, row["provenance"])
        result["assessed" if changed else "stale_skipped"] += 1
    else:
        if len(rows) < limit:
            result["state"] = "caught_up"
    return result


def main(argv=None):
    from .store import Store
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research.sqlite")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)
    store = Store(args.db)
    try:
        print(json.dumps(migrate(store, args.limit), indent=2))
    finally:
        store.db.close()


if __name__ == "__main__":
    main()
