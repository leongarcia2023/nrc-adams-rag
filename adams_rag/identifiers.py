"""Preserve technical identifiers that ordinary word tokenization breaks apart."""
import re

from .storage import require_space

PARSER_VERSION = 1
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})
_TOKEN = re.compile(r"(?<![A-Z0-9])[A-Z0-9]+(?:[./-][A-Z0-9]+)*(?![A-Z0-9])", re.I)
_SHARED_REPORT = re.compile(r"\b(\d{5,8}(?:\s*,\s*\d{5,8})+)/([0-9]{4,8}(?:-[0-9]{1,4})?)\b")


def extract_identifiers(text):
    normalized = re.sub(r"\s*([/-])\s*", r"\1", text.upper().translate(_DASHES))
    found = set()
    for match in _TOKEN.finditer(normalized):
        value = match.group(0)
        digits = sum(c.isdigit() for c in value)
        letters = any("A" <= c <= "Z" for c in value)
        # Avoid ordinary words, isolated years, small decimal quantities and single-digit labels.
        if 4 <= len(value) <= 80 and ((letters and digits >= 2) or (not letters and digits >= 8)):
            found.add(value)
    # NRC findings can combine docket numbers before one report/finding suffix.
    # Expand that explicit notation, without inferring relationships from nearby prose.
    for match in _SHARED_REPORT.finditer(normalized):
        for docket in re.split(r"\s*,\s*", match.group(1)):
            found.add(docket + "/" + match.group(2))
    return found


def index_chunk(db, chunk_id, text):
    db.execute("DELETE FROM chunk_identifiers WHERE chunk_id=?", (chunk_id,))
    db.executemany("INSERT INTO chunk_identifiers(identifier,chunk_id) VALUES(?,?)",
                   [(value, chunk_id) for value in sorted(extract_identifiers(text))])
    db.execute("INSERT INTO identifier_versions VALUES(?,?) ON CONFLICT(chunk_id) DO UPDATE SET version=excluded.version",
               (chunk_id, PARSER_VERSION))


def sync_identifiers(store, limit=2000):
    if not 1 <= limit <= 10000:
        raise ValueError("Use 1–10,000 passages per identifier batch")
    require_space(store.path.parent)
    rows = store.db.execute("""SELECT c.id,c.text FROM chunks c
        LEFT JOIN identifier_versions v ON v.chunk_id=c.id
        WHERE v.chunk_id IS NULL OR v.version<>? ORDER BY c.id LIMIT ?""", (PARSER_VERSION, limit)).fetchall()
    indexed = 0
    for start in range(0, len(rows), 64):
        require_space(store.path.parent)
        with store.db:
            store.db.execute("BEGIN IMMEDIATE")
            for row in rows[start:start + 64]:
                current = store.db.execute("SELECT text FROM chunks WHERE id=?", (row["id"],)).fetchone()
                if current is not None:
                    index_chunk(store.db, row["id"], current[0])
                    indexed += 1
    return indexed
