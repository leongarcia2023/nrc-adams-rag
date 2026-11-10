"""Bounded identifier lookup across immutable catalog observations."""
import base64
from contextlib import ExitStack, closing
import datetime as dt
import hashlib
import json
from pathlib import Path
import zlib

from .catalog_lookup import INDEX, connect, index_ready, normalize_query

MAX_SNAPSHOTS = 64
MAX_MATCHES = 1024


def discover(updates):
    """Return reconciled supplements in observation order; surface unfinished ones."""
    ready, pending = [], []
    for manifest in sorted(Path(updates).glob("*/window.json")):
        folder = manifest.parent
        try:
            window = json.loads(manifest.read_text())
            status = json.loads((folder / "STATUS.json").read_text())
            if not isinstance(window, dict) or not isinstance(status, dict):
                raise ValueError
            if (window.get("scope") != "dated_metadata_supplement" or status.get("window") != window
                    or status.get("state") != "window_count_matched"
                    or not status.get("enumeration_count_matched")):
                raise ValueError
            # STATUS is rewritten on a no-op resume. Use stable reconciliation
            # times so rereading an old window cannot promote it over a new one.
            observed = [dt.datetime.fromisoformat(r["finished"]) for r in status["libraries"]]
            if not observed or any(t.tzinfo is None for t in observed) or not (folder / "catalog.sqlite").is_file():
                raise ValueError
            ready.append((max(observed), str(folder), folder / "catalog.sqlite"))
        except (OSError, ValueError, KeyError, TypeError):
            pending.append(folder.name)
    ready.sort()
    return [item[2] for item in ready], pending


def lookup(base, supplements=(), accession=None, library="both", record_id=None, limit=50, cursor=None):
    """Latest observed metadata wins by NRC identity, not by shared accession.

    Callers order immutable supplements oldest to newest. This is an observation
    merge, not an authority/currentness or withdrawal determination.
    """
    accession, record_id = normalize_query(accession, library, record_id, limit, cursor)
    paths = [Path(p).resolve() for p in (base, *supplements)]
    if len(paths) > MAX_SNAPSHOTS or len(set(paths)) != len(paths):
        raise ValueError("Use distinct catalog snapshots, at most 64; consolidate older supplements first")
    where, params = [], []
    if accession is not None:
        where.append("accession=? COLLATE NOCASE"); params.append(accession)
    if record_id is not None:
        where.append("record_id=?"); params.append(record_id)
    if library == "both":
        where.append("library IN ('main','legacy')")
    else:
        where.append("library=?"); params.append(library)
    table = "items INDEXED BY " + INDEX if accession is not None else "items"
    with ExitStack() as stack:
        databases, snapshots, candidates = [], [], set()
        for order, path in enumerate(paths):
            db = stack.enter_context(closing(connect(path)))
            db.execute("BEGIN")
            libraries = [dict(r) for r in db.execute(
                "SELECT library,final_count,finished,state FROM libraries ORDER BY library")]
            if not libraries or any(r["final_count"] is None or not r["finished"] for r in libraries):
                raise RuntimeError("Catalog observations must be reconciled before unified lookup")
            if accession is not None and not index_ready(db):
                raise RuntimeError("A catalog lookup index is not ready; index every completed supplement first")
            stamp = path.stat()
            identifier = hashlib.sha256(json.dumps([str(path), libraries, stamp.st_size,
                                                    stamp.st_mtime_ns], sort_keys=True).encode()).hexdigest()
            snapshots.append({"snapshot_id": identifier, "order": order,
                              "role": "base" if order == 0 else "supplement", "libraries": libraries})
            databases.append(db)
            rows = db.execute("SELECT library,record_id FROM " + table + " WHERE " +
                              " AND ".join(where) + " LIMIT ?", (*params, MAX_MATCHES + 1)).fetchall()
            candidates.update((r["library"], r["record_id"]) for r in rows)
            if len(candidates) > MAX_MATCHES:
                raise ValueError("Identifier lookup exceeds 1,024 identities; specify an NRC record ID")
        signature = hashlib.sha256(json.dumps([snapshots, accession.casefold() if accession else None,
                                               library, record_id], sort_keys=True).encode()).hexdigest()
        after = None
        if cursor:
            try:
                value = json.loads(base64.b64decode(cursor, altchars=b'-_', validate=True))
                after = value["after"]
                if (value["query"] != signature or not isinstance(after, list) or len(after) != 2
                        or not all(isinstance(v, str) for v in after)):
                    raise ValueError
                after = tuple(after)
            except (ValueError, KeyError, TypeError):
                raise ValueError("Invalid cursor or catalog/query changed; restart the lookup") from None
        matched, page = 0, []
        for identity in sorted(candidates):
            latest, observations = None, []
            for order, db in enumerate(databases):
                row = db.execute("SELECT * FROM items WHERE library=? AND record_id=?", identity).fetchone()
                if row is not None:
                    latest = row
                    observations.append({"snapshot_id": snapshots[order]["snapshot_id"],
                                         "accession": row["accession"]})
            # A changed accession must not leave its older identifier in results.
            if accession is not None and str(latest["accession"] or "").casefold() != accession.casefold():
                continue
            matched += 1
            if (after is None or identity > after) and len(page) <= limit:
                page.append((identity, latest, observations))
        records = []
        for identity, row, observations in page[:limit]:
            metadata = json.loads(zlib.decompress(row["metadata"]))
            for key in ("content", "text", "SearchExcerpt"):
                metadata.pop(key, None)
            records.append({"source_key": f"nrc:{identity[0]}:{identity[1]}",
                "library": identity[0], "record_id": identity[1], "accession": row["accession"],
                "is_package": bool(row["is_package"]), "metadata": metadata,
                "observations": observations, "selected_snapshot": observations[-1]["snapshot_id"],
                "content_scope": "catalog_metadata", "full_text_included": False})
        next_cursor = None
        if len(page) > limit:
            next_cursor = base64.urlsafe_b64encode(json.dumps(
                {"query": signature, "after": page[limit - 1][0]}, separators=(",", ":")).encode()).decode()
        return {"matched_records": matched, "records": records, "next_cursor": next_cursor,
                "snapshot": snapshots[0]["libraries"], "catalog_snapshots": snapshots,
                "selection": "latest_observed_metadata_by_NRC_identity",
                "content_scope": "catalog_metadata", "full_archive_verified": False,
                "withdrawals_detected": False, "current_authority_verified": False}


def lookup_available(base, updates, **query):
    supplements, pending = discover(updates)
    result = lookup(base, supplements, **query)
    result["pending_supplements"] = pending
    result["warnings"] = (["Unfinished or invalid supplements are excluded from this lookup."] if pending else [])
    return result
