from __future__ import annotations

from .source_content import read_text
import hashlib
import json
import re
import sqlite3
import zlib
import uuid
from pathlib import Path
from urllib.parse import urlparse, quote
from .identifiers import PARSER_VERSION, extract_identifiers, index_chunk
from .identity import is_legacy, record_id, source_key
from .page_boundaries import retained_pdf_pages


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def source_url(value, accession):
    """Keep source links navigable without accepting active-content URL schemes."""
    if isinstance(value, str) and urlparse(value).scheme == "https":
        return value
    return "https://adamswebsearch2.nrc.gov/webSearch2/main.jsp?AccessionNumber=" + quote(accession)


def chunk_text(text, size=350, overlap=50):
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between zero and chunk size")
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), size - overlap):
        end = min(start + size, len(words))
        yield text[words[start].start():words[end - 1].end()]
        if end == len(words):
            break


class Store:
    def __init__(self, path):
        from .offload_guard import require_local_data
        require_local_data(path)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        # Reclaim oversized reusable journals after a safe WAL reset. Active
        # readers and uncheckpointed writes may still require a larger WAL.
        self.db.execute("PRAGMA journal_size_limit=67108864")
        self.db.executescript("""
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS documents (
          accession TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
          document_date TEXT, added TEXT, metadata TEXT NOT NULL,
          text_hash TEXT, status TEXT NOT NULL, provenance TEXT NOT NULL,
          updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          accession TEXT NOT NULL REFERENCES documents(accession) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL, page INTEGER, text TEXT NOT NULL,
          UNIQUE(accession, ordinal)
        );
        CREATE TABLE IF NOT EXISTS document_search_keys(
          accession TEXT PRIMARY KEY REFERENCES documents(accession) ON DELETE CASCADE,indexed_accession TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS document_quality(
          accession TEXT PRIMARY KEY REFERENCES documents(accession) ON DELETE CASCADE,
          text_hash TEXT NOT NULL,metadata_hash TEXT NOT NULL,policy_version INTEGER NOT NULL,
          state TEXT NOT NULL,report TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS extraction_reviews(
          accession TEXT NOT NULL, text_hash TEXT NOT NULL, page INTEGER NOT NULL,
          disposition TEXT NOT NULL, reasons TEXT NOT NULL, note TEXT NOT NULL,
          created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY(accession,text_hash,page));
        CREATE TABLE IF NOT EXISTS retrieval_state(id INTEGER PRIMARY KEY CHECK(id=1),generation INTEGER NOT NULL);
        INSERT OR IGNORE INTO retrieval_state VALUES(1,0);
        CREATE TRIGGER IF NOT EXISTS retrieval_documents_insert_v1 AFTER INSERT ON documents
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS retrieval_documents_update_v1 AFTER UPDATE ON documents
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS retrieval_documents_delete_v1 AFTER DELETE ON documents
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS retrieval_chunks_insert_v1 AFTER INSERT ON chunks
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS retrieval_chunks_update_v1 AFTER UPDATE ON chunks
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS retrieval_chunks_delete_v1 AFTER DELETE ON chunks
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS retrieval_reviews_insert_v1 AFTER INSERT ON extraction_reviews
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS retrieval_reviews_update_v1 AFTER UPDATE ON extraction_reviews
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS retrieval_reviews_delete_v1 AFTER DELETE ON extraction_reviews
        BEGIN UPDATE retrieval_state SET generation=generation+1 WHERE id=1; END;
        CREATE TRIGGER IF NOT EXISTS invalidate_text_quality_v1 AFTER UPDATE ON documents
        WHEN old.text_hash IS NOT new.text_hash OR old.metadata IS NOT new.metadata
          OR old.provenance IS NOT new.provenance OR old.status IS NOT new.status
        BEGIN DELETE FROM document_quality WHERE accession=new.accession; END;
        CREATE TABLE IF NOT EXISTS discovery_records(
          library TEXT NOT NULL,record_id TEXT NOT NULL,accession TEXT NOT NULL,
          metadata TEXT NOT NULL,provenance TEXT NOT NULL,updated TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY(library,record_id));
        CREATE INDEX IF NOT EXISTS discovery_accession ON discovery_records(accession,library);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          text, title, accession, content='', tokenize='porter unicode61'
        );
        CREATE TABLE IF NOT EXISTS chunk_identifiers(
          identifier TEXT NOT NULL,chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
          PRIMARY KEY(identifier,chunk_id));
        CREATE INDEX IF NOT EXISTS identifiers_by_chunk ON chunk_identifiers(chunk_id);
        CREATE TABLE IF NOT EXISTS identifier_versions(
          chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS vectors (
          chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
          model TEXT, vector TEXT NOT NULL, PRIMARY KEY(chunk_id, model)
        );
        CREATE TABLE IF NOT EXISTS dense_vectors (
          chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
          model TEXT NOT NULL, dimension INTEGER NOT NULL, vector BLOB NOT NULL,
          content_hash TEXT NOT NULL, PRIMARY KEY(chunk_id,model)
        );
        CREATE TABLE IF NOT EXISTS collection_identity(id INTEGER PRIMARY KEY CHECK(id=1),value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS vector_generations(model TEXT PRIMARY KEY,generation INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS vector_changes(
          model TEXT NOT NULL,chunk_id INTEGER NOT NULL,generation INTEGER NOT NULL,
          PRIMARY KEY(model,chunk_id));
        CREATE INDEX IF NOT EXISTS vector_changes_since ON vector_changes(model,generation);
        DROP TRIGGER IF EXISTS vectors_insert_generation;
        DROP TRIGGER IF EXISTS vectors_delete_generation;
        DROP TRIGGER IF EXISTS vectors_update_generation;
        CREATE TRIGGER IF NOT EXISTS vectors_insert_generation_v2 AFTER INSERT ON dense_vectors BEGIN
          INSERT INTO vector_generations VALUES(new.model,1)
          ON CONFLICT(model) DO UPDATE SET generation=generation+1;
          INSERT INTO vector_changes VALUES(new.model,new.chunk_id,(SELECT generation FROM vector_generations WHERE model=new.model))
          ON CONFLICT(model,chunk_id) DO UPDATE SET generation=excluded.generation;
        END;
        CREATE TRIGGER IF NOT EXISTS vectors_delete_generation_v2 AFTER DELETE ON dense_vectors BEGIN
          INSERT INTO vector_generations VALUES(old.model,1)
          ON CONFLICT(model) DO UPDATE SET generation=generation+1;
          INSERT INTO vector_changes VALUES(old.model,old.chunk_id,(SELECT generation FROM vector_generations WHERE model=old.model))
          ON CONFLICT(model,chunk_id) DO UPDATE SET generation=excluded.generation;
        END;
        CREATE TRIGGER IF NOT EXISTS vectors_update_generation_v2 AFTER UPDATE ON dense_vectors BEGIN
          INSERT INTO vector_generations VALUES(old.model,1)
          ON CONFLICT(model) DO UPDATE SET generation=generation+1;
          INSERT INTO vector_changes VALUES(old.model,old.chunk_id,(SELECT generation FROM vector_generations WHERE model=old.model))
          ON CONFLICT(model,chunk_id) DO UPDATE SET generation=excluded.generation;
          INSERT INTO vector_generations VALUES(new.model,1)
          ON CONFLICT(model) DO UPDATE SET generation=generation+1;
          INSERT INTO vector_changes VALUES(new.model,new.chunk_id,(SELECT generation FROM vector_generations WHERE model=new.model))
          ON CONFLICT(model,chunk_id) DO UPDATE SET generation=excluded.generation;
        END;
        CREATE TABLE IF NOT EXISTS document_versions (
          accession TEXT NOT NULL, version TEXT NOT NULL, text_hash TEXT NOT NULL,
          metadata TEXT NOT NULL, content BLOB NOT NULL, provenance TEXT NOT NULL,
          captured TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(accession,version)
        );
        CREATE TABLE IF NOT EXISTS datasets (
          id TEXT PRIMARY KEY, manifest TEXT NOT NULL, imported TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS crawl (
          job TEXT, day TEXT, offset INTEGER DEFAULT 0, state TEXT DEFAULT 'pending',
          last_page_hash TEXT, note TEXT, PRIMARY KEY(job, day)
        );
        CREATE TABLE IF NOT EXISTS http_state (
          scope TEXT PRIMARY KEY, next_at REAL DEFAULT 0, day TEXT, used INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS retrieval_events (
          id INTEGER PRIMARY KEY, accession TEXT, outcome TEXT,
          recorded TEXT DEFAULT CURRENT_TIMESTAMP
        );
        COMMIT;
        """)
        # Progress, quality-version checks and the old search-excerpt migration
        # must not read every document's full metadata. These small indexes also
        # keep frequent worker restarts independent of that growing text.
        for name, columns in (
            ("documents_status_provenance", "status,provenance"),
            ("documents_source_version", "accession,text_hash"),
        ):
            if not self.db.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (name,)).fetchone():
                from .storage import require_space
                count = self.db.execute("SELECT count(*) FROM documents").fetchone()[0]
                require_space(path.parent, extra_bytes=max(32 * 1024 * 1024, count * 256))
                with self.db:
                    self.db.execute(f"CREATE INDEX IF NOT EXISTS {name} ON documents({columns})")
        # Older prototype versions indexed search snippets. Remove those from evidence.
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO collection_identity VALUES(1,?)", (uuid.uuid4().hex,))
            if not self.db.execute("SELECT 1 FROM vector_generations LIMIT 1").fetchone():
                self.db.execute("INSERT OR IGNORE INTO vector_generations SELECT DISTINCT model,0 FROM dense_vectors")
            self.db.execute("""UPDATE crawl SET state='obsolete_filter',
              note='Old date equality could omit daytime records; rerun this interval with the range-filter version'
              WHERE json_extract(job,'$.date_filter_version') IS NULL""")
            for old in self.db.execute("SELECT accession,title FROM documents WHERE provenance='api_search_unverified' AND status='indexed'").fetchall():
                for chunk in self.db.execute("SELECT id,text FROM chunks WHERE accession=?", (old["accession"],)).fetchall():
                    self.db.execute("INSERT INTO chunks_fts(chunks_fts,rowid,text,title,accession) VALUES('delete',?,?,?,?)",
                                    (chunk["id"], chunk["text"], old["title"], old["accession"]))
                self.db.execute("DELETE FROM chunks WHERE accession=?", (old["accession"],))
                self.db.execute("UPDATE documents SET status='search_excerpt',provenance='api_search_excerpt' WHERE accession=?", (old["accession"],))

    def put_discovery(self, document, provenance="api_search_excerpt"):
        identity = record_id(document)
        library = "legacy" if is_legacy(document) else "main"
        if library == "legacy":
            source_key(document)  # Require an unambiguous NRC identity.
        metadata = dict(document)
        for key in ("content", "text", "SearchExcerpt"):
            metadata.pop(key, None)
        self.db.execute("""INSERT INTO discovery_records(library,record_id,accession,metadata,provenance)
            VALUES(?,?,?,?,?) ON CONFLICT(library,record_id) DO UPDATE SET
            accession=excluded.accession,metadata=excluded.metadata,provenance=excluded.provenance,updated=CURRENT_TIMESTAMP""",
            (library, identity, document["AccessionNumber"], json.dumps(metadata), provenance))

    def put(self, document, provenance="import", pages=None):
        """Participates in caller transaction, allowing atomic page/checkpoint commits."""
        accession = document.get("AccessionNumber") or document.get("_id")
        if not isinstance(accession, str) or not accession.strip():
            raise ValueError("Document has no accession/identifier")
        public_accession = accession
        legacy = is_legacy(document)
        if provenance.startswith("api_search"):
            self.put_discovery(document, provenance)
            if legacy:
                return "legacy_metadata"  # Search excerpts cannot establish full-document content.
        if legacy:
            accession = source_key(document)
        title = str(document.get("DocumentTitle") or document.get("title") or document.get("Name") or public_accession)
        body = document.get("content", document.get("text", "")) or ""
        if not isinstance(body, str):
            raise ValueError(f"Unsupported content encoding for {accession}; retain and normalize before import")
        if document.get("contentEncoding") == "base64":
            import base64
            body = base64.b64decode(body, validate=True).decode("utf-8")
        if pages is not None:
            body = "\f".join(pages)
        from .text_quality import is_unavailable_notice
        availability_notice = body if is_unavailable_notice(body) else None
        if availability_notice:
            body = ""
        package = str(document.get("IsPackage", "")).lower() in {"yes", "true", "1"}
        status = "package" if package else ("indexed" if body.strip() else "missing_text")
        if legacy and status == "missing_text":
            status = "legacy_metadata"
        search_excerpt = provenance.startswith("api_search")
        if search_excerpt and status == "indexed":
            status = "search_excerpt"
        metadata = dict(document)
        metadata.pop("content", None)
        metadata.pop("text", None)
        if availability_notice:
            metadata["SourceContentAvailabilityNotice"] = availability_notice
        if legacy:
            metadata.update(SourceAccession=public_accession, SourceLibrary="legacy", SourceRecordId=record_id(document))
        if search_excerpt:
            metadata["SearchExcerpt"] = body
        old = self.db.execute("SELECT * FROM documents WHERE accession=?", (accession,)).fetchone()
        # An API metadata refresh must not erase previously imported source text or page locations.
        preserve = old and old["status"] == "indexed" and not package and (
            search_excerpt or not body.strip() or (old["provenance"] == "pdf" and provenance != "pdf")
        )
        text_hash = old["text_hash"] if preserve else digest(body)
        if preserve:
            status, provenance = old["status"], old["provenance"]
            previous_metadata = json.loads(old["metadata"])
            old_page_lengths = previous_metadata.get("RetainedPageLengths")
            previous_metadata.update(metadata)
            metadata = previous_metadata
            # Discovery refreshes cannot replace the physical map of retained text.
            if old_page_lengths is not None:
                metadata["RetainedPageLengths"] = old_page_lengths
            else:
                metadata.pop("RetainedPageLengths", None)
            if not any(document.get(key) for key in ("DocumentTitle", "title", "Name")):
                title = old["title"]
        if not preserve and provenance == "pdf" and pages is not None:
            metadata["RetainedPageLengths"] = [len(page) for page in pages]
            retained_pdf_pages(body, metadata)  # Reject contradictory physical counts before publication.
        page_layout_changed = bool(old and not preserve and pages is not None and provenance == "pdf"
            and json.loads(old["metadata"]).get("RetainedPageLengths") != metadata.get("RetainedPageLengths"))
        if status == "indexed" and not preserve:
            serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
            self.db.execute("INSERT OR IGNORE INTO document_versions(accession,version,text_hash,metadata,content,provenance) VALUES(?,?,?,?,?,?)",
                (accession, digest(text_hash + serialized), text_hash, serialized, zlib.compress(body.encode()), provenance))
        self.db.execute("""INSERT INTO documents
            (accession,title,url,document_date,added,metadata,text_hash,status,provenance)
            VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(accession) DO UPDATE SET
            title=excluded.title,url=excluded.url,document_date=excluded.document_date,
            added=excluded.added,metadata=excluded.metadata,text_hash=excluded.text_hash,
            status=excluded.status,provenance=excluded.provenance,updated=CURRENT_TIMESTAMP""",
            (accession, title, source_url(metadata.get("Url") or (old["url"] if preserve else None), public_accession),
             metadata.get("DocumentDate"), metadata.get("DateAddedTimestamp"),
             json.dumps(metadata, ensure_ascii=False), text_hash, status, provenance))
        from .text_quality import POLICY_VERSION as QUALITY_VERSION, assess, classify, save
        previous_quality = self.db.execute("SELECT report FROM document_quality WHERE accession=? AND text_hash=? AND policy_version=?",
                                          (accession, text_hash, QUALITY_VERSION)).fetchone() if preserve else None
        if previous_quality:
            quality = classify(json.loads(previous_quality[0]), metadata, provenance, status)
        elif preserve:
            retained = self.db.execute("SELECT content,metadata FROM document_versions WHERE accession=? AND text_hash=? ORDER BY captured DESC,rowid DESC LIMIT 1",
                                       (accession, text_hash)).fetchone()
            retained_text = read_text(retained[0], self.db) if retained else ""
            quality = assess(retained_text, metadata, provenance, status,
                             retained_pdf_pages(retained_text, json.loads(retained[1])) if retained and provenance == "pdf" else None,
                             text_available=bool(retained))
        else:
            quality = assess(body, metadata, provenance, status, pages)
        save(self.db, accession, text_hash, json.dumps(metadata, ensure_ascii=False), quality, provenance)
        saved_key = self.db.execute("SELECT indexed_accession FROM document_search_keys WHERE accession=?", (accession,)).fetchone()
        old_search_key = saved_key[0] if saved_key else accession
        search_key = accession + " " + public_accession if legacy else accession
        if old and old["text_hash"] == text_hash and old["title"] == title and old["status"] == status and old["provenance"] == provenance and old_search_key == search_key and not page_layout_changed:
            return status
        self.db.execute("INSERT INTO document_search_keys VALUES(?,?) ON CONFLICT(accession) DO UPDATE SET indexed_accession=excluded.indexed_accession",
                        (accession, search_key))
        old_chunks = self.db.execute("SELECT id,text FROM chunks WHERE accession=?", (accession,)).fetchall()
        if preserve:
            # Only the title changed. Rebuild the FTS entries without losing page positions/vectors.
            for c in old_chunks:
                self.db.execute("INSERT INTO chunks_fts(chunks_fts,rowid,text,title,accession) VALUES('delete',?,?,?,?)",
                                (c["id"], c["text"], old["title"], old_search_key))
                self.db.execute("INSERT INTO chunks_fts(rowid,text,title,accession) VALUES(?,?,?,?)",
                                (c["id"], c["text"], title, search_key))
                self.db.execute("DELETE FROM vectors WHERE chunk_id=?", (c["id"],))
                self.db.execute("DELETE FROM dense_vectors WHERE chunk_id=?", (c["id"],))
            return status
        for c in old_chunks:
            self.db.execute("INSERT INTO chunks_fts(chunks_fts,rowid,text,title,accession) VALUES('delete',?,?,?,?)",
                            (c["id"], c["text"], old["title"], old_search_key))
        self.db.execute("DELETE FROM chunks WHERE accession=?", (accession,))
        if status != "indexed":
            return status
        ordinal = 0
        for page, text in (enumerate(pages, 1) if pages is not None else [(None, body)]):
            for chunk in chunk_text(text):
                ordinal += 1
                cursor = self.db.execute("INSERT INTO chunks(accession,ordinal,page,text) VALUES(?,?,?,?)",
                                         (accession, ordinal, page, chunk))
                self.db.execute("INSERT INTO chunks_fts(rowid,text,title,accession) VALUES(?,?,?,?)",
                                (cursor.lastrowid, chunk, title, search_key))
                index_chunk(self.db, cursor.lastrowid, chunk)
        return status

    def status(self):
        from .text_quality import POLICY_VERSION as QUALITY_VERSION
        has_recovery = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='pdf_recovery'").fetchone()
        has_markdown = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='markdown_checks'").fetchone()
        return {
            "markdown_checks": dict(self.db.execute('SELECT state,count(*) FROM markdown_checks GROUP BY state')) if has_markdown else {},
            "documents": dict(self.db.execute("SELECT status,count(*) FROM documents GROUP BY status")),
            "discovery_records": dict(self.db.execute("SELECT library,count(*) FROM discovery_records GROUP BY library")),
            "chunks": self.db.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "identifier_passages": self.db.execute("SELECT count(*) FROM identifier_versions WHERE version=?", (PARSER_VERSION,)).fetchone()[0],
            "vectors_by_model": dict(self.db.execute("SELECT model,count(*) FROM vectors GROUP BY model")),
            "dense_vectors_by_model": dict(self.db.execute("SELECT model,count(*) FROM dense_vectors GROUP BY model")),
            "document_versions": self.db.execute("SELECT count(*) FROM document_versions").fetchone()[0],
            "text_quality": dict(self.db.execute("SELECT q.state,count(*) FROM document_quality q JOIN documents d USING(accession) WHERE q.text_hash=d.text_hash AND q.policy_version=? GROUP BY q.state", (QUALITY_VERSION,))),
            "pdf_recovery": dict(self.db.execute("SELECT outcome,count(*) FROM pdf_recovery GROUP BY outcome")) if has_recovery else {},
            "datasets": [json.loads(r[0]) for r in self.db.execute("SELECT manifest FROM datasets")],
            "crawl_partitions": dict(self.db.execute("SELECT state,count(*) FROM crawl GROUP BY state")),
            "full_archive_verified": False,
        }

    @staticmethod
    def _retrieval_filters(docket=None, date_from=None, date_to=None):
        where, params = "", []
        if docket:
            where += " AND EXISTS(SELECT 1 FROM json_each(d.metadata, '$.DocketNumber') WHERE value=?)"
            params.append(docket)
        if date_from:
            where += " AND substr(d.document_date,1,10)>=?"
            params.append(date_from)
        if date_to:
            where += " AND substr(d.document_date,1,10)<=?"
            params.append(date_to)
        return where, params

    def opening_chunks(self, accessions, docket=None, date_from=None, date_to=None):
        where, filters = self._retrieval_filters(docket, date_from, date_to)
        rows = []
        for accession in sorted(set(accessions))[:8]:
            row = self.db.execute("""SELECT c.*,d.title,d.url,d.provenance,d.document_date
                FROM chunks c JOIN documents d ON d.accession=c.accession
                WHERE c.accession=?""" + where + " ORDER BY c.ordinal LIMIT 1", (accession, *filters)).fetchone()
            if row:
                rows.append(dict(row))
        return rows

    def identifier_matches(self, question, limit=100, docket=None, date_from=None, date_to=None):
        identifiers = sorted(value for value in extract_identifiers(question)
                             if not re.fullmatch(r"ML\d{5}[A-Z0-9]\d{3}", value))[:12]
        if not identifiers:
            return []
        where, filters = self._retrieval_filters(docket, date_from, date_to)
        placeholders = ",".join("?" for _ in identifiers)
        expression = " OR ".join(filter(None, [self._fts_expression(question), self._fts_expression(" ".join(identifiers))]))
        # Drive FTS lookups from the exact identifiers. An ordinary join lets
        # SQLite scan every broad text match before checking this small set.
        return [dict(row) for row in self.db.execute("""WITH exact AS MATERIALIZED (
            SELECT chunk_id,count(*) AS identifier_hits FROM chunk_identifiers
            WHERE identifier IN (""" + placeholders + """ ) GROUP BY chunk_id)
            SELECT c.*,d.title,d.url,d.provenance,d.document_date,d.metadata,exact.identifier_hits
            FROM exact CROSS JOIN chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
            JOIN documents d ON d.accession=c.accession
            WHERE chunks_fts.rowid=exact.chunk_id AND chunks_fts MATCH ?""" + where +
            " ORDER BY identifier_hits DESC,bm25(chunks_fts,1.0,2.0,4.0) LIMIT ?",
            (*identifiers, expression, *filters, limit))]

    @staticmethod
    def _fts_expression(question):
        tokens = list(dict.fromkeys(re.findall(r"[A-Za-z0-9]+", question)))[:40]
        stop = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "what", "when", "where", "which", "with", "did", "does", "how"}
        tokens = [t for t in tokens if t.lower() not in stop]
        return " OR ".join('"' + t + '"' for t in tokens)

    def lexical(self, question, limit=30, docket=None, date_from=None, date_to=None, *, title_only=False):
        from .query_terms import subject_phrases, phrase_expression
        phrases = subject_phrases(question) if not re.search(r'\bML\d{5}[A-Z0-9]\d{3}\b', question, re.I) else []
        expression = phrase_expression(phrases) if phrases else self._fts_expression(question)
        if not expression:
            return []
        if title_only:
            expression = 'title : (' + expression + ')'
        where, filters = self._retrieval_filters(docket, date_from, date_to)
        params = [expression, *filters, limit]
        # FTS5 can return weighted BM25 matches in rank order. Apply filters
        # before LIMIT, then load passage bodies only for the selected rows.
        return [dict(row) for row in self.db.execute("""WITH ranked AS MATERIALIZED (
          SELECT c.id AS chunk_id,chunks_fts.rank AS score
          FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
          JOIN documents d ON d.accession=c.accession
          WHERE chunks_fts MATCH ? AND rank MATCH 'bm25(1.0,2.0,4.0)'""" + where + """
          ORDER BY rank LIMIT ?)
          SELECT c.*,d.title,d.url,d.provenance,d.document_date,ranked.score
          FROM ranked JOIN chunks c ON c.id=ranked.chunk_id
          JOIN documents d ON d.accession=c.accession
          ORDER BY ranked.score""", params)]
