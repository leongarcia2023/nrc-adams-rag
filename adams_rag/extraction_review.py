"""Version-bound observations of extraction errors; original evidence stays retained."""
import json

from .storage import require_space

REASONS = {"numeric_ocr_error", "table_layout_loss", "omitted_text", "incorrect_text",
           "unrepresented_visual_content"}


def record(store, accession, text_hash, page, disposition, reasons, note):
    """Record a concrete review against the exact text inspected, without rewriting it.

    Page 0 applies to the whole document. Existing observations are immutable;
    a new extraction has a different hash and must be evaluated separately.
    """
    if (type(page) is not int or page < 0 or disposition not in {"warn", "exclude"}
            or not isinstance(reasons, list) or not reasons
            or any(not isinstance(reason,str) or reason not in REASONS for reason in reasons)
            or not isinstance(note, str) or not 1 <= len(note.strip()) <= 2000):
        raise ValueError("Use a document/page, review disposition, supported reasons and a concise observation")
    require_space(store.path.parent)
    with store.db:
        store.db.execute("BEGIN IMMEDIATE")
        document = store.db.execute("SELECT text_hash,status FROM documents WHERE accession=?", (accession,)).fetchone()
        if not document or document["status"] != "indexed" or document["text_hash"] != text_hash:
            raise ValueError("The document changed or is unavailable; review its current extraction first")
        if page and not store.db.execute("SELECT 1 FROM chunks WHERE accession=? AND page=? LIMIT 1", (accession,page)).fetchone():
            raise ValueError("The reviewed physical page has no retained passages")
        values = (disposition, json.dumps(sorted(set(reasons))), note.strip())
        old = store.db.execute("SELECT disposition,reasons,note FROM extraction_reviews WHERE accession=? AND text_hash=? AND page=?",
                               (accession,text_hash,page)).fetchone()
        if old:
            if tuple(old) != values:
                raise ValueError("An existing extraction review is immutable")
            return False
        store.db.execute("INSERT INTO extraction_reviews(accession,text_hash,page,disposition,reasons,note) VALUES(?,?,?,?,?,?)",
                         (accession,text_hash,page,*values))
    return True


def for_version(db, accession, text_hash):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_reviews'").fetchone():
        return []
    return [{**dict(r), "reasons": json.loads(r["reasons"])} for r in db.execute(
        "SELECT * FROM extraction_reviews WHERE accession=? AND text_hash=? ORDER BY page", (accession,text_hash))]


def annotate(quality, db, accession, text_hash):
    reviews = for_version(db, accession, text_hash)
    if reviews:
        quality = {**quality, "state": "needs_review", "evidence_reviews": reviews,
                   "flags": list(dict.fromkeys(quality.get("flags",[]) + ["reviewed_extraction_issue"]))}
    return quality


def filter_rows(db, rows, exclusions=None):
    """Apply the same exclusion after lexical, identifier and semantic retrieval."""
    from .markdown_archive import filter_rows as filter_markdown
    rows = filter_markdown(db, rows, exclusions)
    if not rows or not db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_reviews'").fetchone():
        return rows
    ids = list(dict.fromkeys(row["id"] for row in rows))
    blocked = set()
    reported = {(item['accession'],item['text_hash'],item['page']) for item in (exclusions or [])}
    for start in range(0,len(ids),400):
        batch = ids[start:start + 400]
        placeholders = ",".join("?" for _ in batch)
        matches = db.execute("""SELECT c.id AS chunk_id,r.* FROM chunks c
            JOIN documents d ON d.accession=c.accession
            JOIN extraction_reviews r ON r.accession=d.accession AND r.text_hash=d.text_hash
            WHERE r.disposition='exclude' AND (r.page=0 OR r.page=c.page)
            AND c.id IN (""" + placeholders + ")", batch)
        for row in matches:
            blocked.add(row["chunk_id"])
            identity = (row["accession"],row["text_hash"],row["page"])
            if exclusions is not None and identity not in reported:
                reported.add(identity)
                exclusions.append({k:json.loads(row[k]) if k == "reasons" else row[k]
                                   for k in ("accession","text_hash","page","disposition","reasons","note")})
    return [row for row in rows if row["id"] not in blocked]
