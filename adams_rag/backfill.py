"""Bounded full-text ingestion directly from the reconciled catalog, with durable cursors."""
import hashlib
import fcntl
import json
from pathlib import Path
import re
import sqlite3
import zlib

from .client import AdamsHTTPError, BudgetStop
from .ingest import archive
from .research import ACCESSION
from .storage import StoragePause, require_space, reserve_gb
from .identity import is_legacy, record_id as nrc_record_id, source_key

# Older legacy cursors only recorded pending metadata. Never reuse those
# checkpoints for actual acquisition or count them as attempted text requests.
LEGACY_ACQUISITION_VERSION = 1


def backfill(store, client, catalog, max_documents=20, library="main", min_free_gb=None, progress=None, cancelled=None):
    # CLI, continuous worker and other callers must not advance the same cursor together.
    with store.path.with_suffix(".backfill.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another full-text backfill is already running") from None
        return _backfill(store, client, catalog, max_documents, library, min_free_gb, progress, cancelled)


def _backfill(store, client, catalog, max_documents, library, min_free_gb, progress, cancelled):
    min_free_gb = reserve_gb(min_free_gb)
    if not 1 <= max_documents <= 10000 or library not in {"main", "legacy"}:
        raise ValueError("Use 1–10,000 documents and a public library")
    path = Path(catalog).resolve()
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        snapshot = source.execute("SELECT library,final_count,finished,state FROM libraries WHERE library=?", (library,)).fetchone()
        if not snapshot or not snapshot["finished"]:
            raise ValueError("Backfill requires a reconciled catalog snapshot")
        signature = [str(path), dict(snapshot)]
        if library == "legacy":
            signature.append({"legacy_acquisition_version": LEGACY_ACQUISITION_VERSION})
        job = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        store.db.executescript("""CREATE TABLE IF NOT EXISTS backfill_cursors(
          job TEXT PRIMARY KEY, library TEXT NOT NULL, catalog TEXT NOT NULL,
          last_row INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending',
          updated TEXT DEFAULT CURRENT_TIMESTAMP);
          CREATE TABLE IF NOT EXISTS backfill_records(
            job TEXT NOT NULL, catalog_row INTEGER NOT NULL, accession TEXT NOT NULL,
            outcome TEXT NOT NULL, detail TEXT, record_id TEXT, updated TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(job,catalog_row));
          CREATE TABLE IF NOT EXISTS backfill_totals(
            job TEXT NOT NULL, outcome TEXT NOT NULL, count INTEGER NOT NULL,
            PRIMARY KEY(job,outcome));
        """)
        if "record_id" not in {r[1] for r in store.db.execute("PRAGMA table_info(backfill_records)")}:
            store.db.execute("ALTER TABLE backfill_records ADD COLUMN record_id TEXT")
        with store.db:
            store.db.execute("INSERT OR IGNORE INTO backfill_cursors(job,library,catalog) VALUES(?,?,?)", (job, library, str(path)))
        position = store.db.execute("SELECT last_row FROM backfill_cursors WHERE job=?", (job,)).fetchone()[0]
        result = {"job": job, "library": library, "attempted": 0, "indexed": 0, "missing_text": [],
                  "already_indexed": 0, "unsupported_accessions": 0, "unavailable": 0,
                  "legacy_metadata": 0, "legacy_identity_mismatch": 0,
                  "examined": 0, "state": "batch_limit", "warnings": []}
        try:
            require_space(store.path.parent, min_free_gb)
        except StoragePause as error:
            with store.db:
                store.db.execute("UPDATE backfill_cursors SET state='storage_pause',updated=CURRENT_TIMESTAMP WHERE job=?", (job,))
            result.update(state="storage_pause", warnings=[str(error)], last_catalog_row=position, full_archive_verified=False)
            return result
        # Upgrade older audit rows without changing the acquisition cursor.
        upgrades = []
        for old in store.db.execute("SELECT catalog_row FROM backfill_records WHERE job=? AND record_id IS NULL LIMIT 1000", (job,)).fetchall():
            previous = source.execute("SELECT metadata FROM items WHERE rowid=? AND library=?", (old[0], library)).fetchone()
            if previous:
                upgrades.append((nrc_record_id(json.loads(zlib.decompress(previous[0]))), job, old[0]))
        with store.db:
            store.db.executemany("UPDATE backfill_records SET record_id=? WHERE job=? AND catalog_row=?", upgrades)
        max_examined = max(500, max_documents * 20)
        def checkpoint(rowid, state="running"):
            store.db.execute("UPDATE backfill_cursors SET last_row=?,state=?,updated=CURRENT_TIMESTAMP WHERE job=?", (rowid, state, job))
        def record(row, outcome, detail=None):
            identity = nrc_record_id(json.loads(zlib.decompress(row["metadata"])))
            inserted = store.db.execute("INSERT OR IGNORE INTO backfill_records(job,catalog_row,accession,outcome,detail,record_id) VALUES(?,?,?,?,?,?)",
                                       (job, row["rowid"], row["accession"], outcome, detail, identity)).rowcount
            if inserted:
                store.db.execute("INSERT INTO backfill_totals VALUES(?,?,1) ON CONFLICT(job,outcome) DO UPDATE SET count=count+1", (job, outcome))
            checkpoint(row["rowid"])
        while result["attempted"] < max_documents and result["examined"] < max_examined:
            if cancelled and cancelled():
                result["state"] = "stopped"
                break
            # The catalog's library index otherwise sorts millions of rows for each
            # record. Scan the immutable rowid sequence with a durable lower bound.
            row = source.execute("SELECT rowid,accession,metadata FROM items NOT INDEXED WHERE library=? AND is_package=0 AND rowid>? ORDER BY rowid LIMIT 1", (library, position)).fetchone()
            if row is None:
                result["state"] = "catalog_exhausted"
                break
            try:
                require_space(store.path.parent, min_free_gb)
            except StoragePause as error:
                result["state"] = "storage_pause"
                result["warnings"].append(str(error))
                break
            accession = row["accession"]
            catalog_metadata = json.loads(zlib.decompress(row["metadata"]))
            identity = nrc_record_id(catalog_metadata)
            storage_key = accession
            supported = bool(ACCESSION.fullmatch(accession))
            if library == "legacy":
                if not is_legacy(catalog_metadata):
                    raise ValueError("Legacy catalog record is missing its library identity")
                storage_key = source_key(catalog_metadata)
                supported = bool(re.fullmatch(r"\d{10}", accession))
            result["examined"] += 1
            old = store.db.execute("SELECT status FROM documents WHERE accession=?", (storage_key,)).fetchone()
            if not supported or (old and old[0] == "indexed"):
                result["already_indexed" if old and old[0] == "indexed" else "unsupported_accessions"] += 1
                with store.db:
                    record(row, "already_indexed" if old and old[0] == "indexed" else "unsupported_accession")
                position = row["rowid"]
                continue
            if progress:
                progress(accession, result["attempted"], max_documents)
            try:
                response = client.request(accession=accession)
            except AdamsHTTPError as error:
                if error.status not in {404, 410}:
                    raise  # Authentication or schema failures must not skip the collection.
                result["attempted"] += 1
                result["unavailable"] += 1
                with store.db:
                    record(row, "unavailable", f"HTTP {error.status}; eligible for later retry")
                position = row["rowid"]
                continue
            except BudgetStop as error:
                result["state"] = "budget_pause"
                result["warnings"].append(str(error))
                break  # Do not advance past the unattempted record.
            except StoragePause as error:
                result["state"] = "storage_pause"
                result["warnings"].append(str(error))
                break
            result["attempted"] += 1
            doc = response.get("document") if isinstance(response, dict) else None
            if not isinstance(doc, dict) or doc.get("AccessionNumber", "").upper() != accession.upper():
                raise ValueError("ADAMS returned an unexpected document identifier; backfill cursor retained")
            if library == "main" and is_legacy(doc):
                raise ValueError("ADAMS returned a legacy record for a main-library request; backfill cursor retained")
            if library == "legacy":
                if not is_legacy(doc) or not nrc_record_id(doc).startswith("id:"):
                    raise ValueError("ADAMS legacy response lacks a matching library or NRC record identity; cursor retained")
            metadata = dict(catalog_metadata)
            metadata.update(doc)
            # Reserve for chunk copies, FTS, versions and WAL before storing a large response.
            try:
                require_space(store.path.parent, min_free_gb,
                              extra_bytes=max(32 * 1024 * 1024, len(json.dumps(response).encode()) * 12))
            except StoragePause as error:
                result["state"] = "storage_pause"
                result["warnings"].append(str(error))
                break  # The record is fetched again after storage is available.
            # A hint only: the next loop still checks eligibility, response
            # identity and storage before committing that record in order.
            # Never look past this batch or fetch known indexed/unsupported rows.
            if (library == "main" and hasattr(client, "prefetch")
                    and result["attempted"] < max_documents and result["examined"] < max_examined
                    and not (cancelled and cancelled())):
                slots = min(getattr(client, "lookahead_depth", 1), max_documents - result["attempted"],
                            max_examined - result["examined"])
                following_rows = source.execute("SELECT rowid,accession FROM items NOT INDEXED WHERE library=? AND is_package=0 AND rowid>? ORDER BY rowid LIMIT ?", (library, row["rowid"], slots)).fetchall()
                for following in following_rows:
                    if not ACCESSION.fullmatch(following["accession"]):
                        continue
                    indexed = store.db.execute("SELECT status FROM documents WHERE accession=?", (following["accession"],)).fetchone()
                    if not indexed or indexed[0] != "indexed":
                        client.prefetch(following["accession"])
            archive(store, response)
            if library == "legacy" and nrc_record_id(doc) != identity:
                # GetDocument can select a sibling sharing the accession. Keep
                # the response for audit, but never assign its text to this row.
                with store.db:
                    record(row, "legacy_identity_mismatch", json.dumps({
                        "requested_record_id": identity, "returned_record_id": nrc_record_id(doc),
                        "eligible_for_identity_specific_retry": True}))
                result["legacy_identity_mismatch"] += 1
                position = row["rowid"]
                continue
            if library == "legacy":
                metadata.update(SourceContentScope="legacy_indexed_text_unverified",
                                SourceContentEndpoint="GetDocument",
                                SourceRecordIdentityMatched=True)
            with store.db:
                state = store.put(metadata, "api_document_unverified")
                if state == "missing_text":
                    store.db.execute("UPDATE documents SET status='unavailable_text' WHERE accession=?", (accession,))
                store.db.execute("INSERT INTO retrieval_events(accession,outcome) VALUES(?,?)", (storage_key, state))
                record(row, state)
            if state == "indexed":
                result["indexed"] += 1
            elif state == "legacy_metadata":
                result["legacy_metadata"] += 1
                result["missing_text"].append(accession)
            elif state != "package":
                result["missing_text"].append(accession)
            position = row["rowid"]
        with store.db:
            checkpoint(position, result["state"])
        result.update(last_catalog_row=position, full_archive_verified=False)
        return result
    finally:
        source.close()
