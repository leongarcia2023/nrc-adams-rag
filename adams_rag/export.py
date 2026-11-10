"""Portable Markdown views of retained source text; checked Markdown is the canonical text store."""
from .source_content import read_text
import json
from .store import source_url
from .page_boundaries import retained_pdf_pages


def document_markdown(store, accession, version=None):
    from .markdown_archive import saved_markdown, decode_markdown
    saved = saved_markdown(store, accession)
    if saved is not None:
        if version is None or decode_markdown(saved)[0]['retained_version'] == version:
            return saved
    document = store.db.execute("SELECT * FROM documents WHERE accession=?", (accession,)).fetchone()
    if not document:
        raise ValueError("Document is not in the local collection")
    if version:
        row = store.db.execute("SELECT * FROM document_versions WHERE accession=? AND version=?", (accession, version)).fetchone()
    else:
        row = store.db.execute("SELECT * FROM document_versions WHERE accession=? AND text_hash=? ORDER BY captured DESC,rowid DESC LIMIT 1", (accession, document["text_hash"])).fetchone()
    if not row:
        raise ValueError("No retained full-text version is available for this document")
    metadata = json.loads(row["metadata"])
    body = read_text(row["content"], store.db)
    pages = retained_pdf_pages(body, metadata) if row["provenance"] == "pdf" else None
    from .text_quality import assess
    quality = assess(body, metadata, row["provenance"], "indexed",
                     pages)
    from .extraction_review import annotate
    quality = annotate(quality, store.db, accession, row["text_hash"])
    title = metadata.get("DocumentTitle") or metadata.get("title") or document["title"]
    header = {"accession": metadata.get("SourceAccession", accession), "source_key": accession,
              "nrc_record_id": metadata.get("SourceRecordId"), "title": title,
              "source_url": source_url(metadata.get("Url") or document["url"], metadata.get("SourceAccession", accession)),
              "document_date": metadata.get("DocumentDate"), "provenance": row["provenance"],
              "version": row["version"], "text_sha256": row["text_hash"], "captured_utc": row["captured"],
              "dataset": metadata.get("Dataset"), "extraction_verified": False, "text_quality": quality}
    if row["provenance"] == "pdf":
        header["source_pdf_sha256"] = metadata.get("LocalFileSHA256")
        reviewed = [{key: page.get(key) for key in ("page", "correction_id", "reviewer", "reviewed_utc", "review_note", "source_pdf_sha256")}
                    for page in metadata.get("PageExtraction", []) if page.get("method") == "reviewed_transcription"]
        if reviewed:
            header["reviewed_page_transcriptions"] = reviewed
    text = "---\n" + "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in header.items()) + "\n---\n\n"
    text += "# " + " ".join(str(title).split()) + "\n\n"
    if row["provenance"] == "pdf":
        for number, page in enumerate(pages, 1):
            text += f"## PDF page {number}\n\n{page.strip()}\n\n"
    else:
        text += body.strip() + "\n"
    return text
