"""Enumerate a dated supplement without changing the reconciled base catalog."""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import signal

from .catalog import Catalog, atomic_write
from .client import BudgetStop, TemporaryRequestError, unpack_search
from .identity import is_legacy
from .storage import StoragePause, require_space


def window(start, end):
    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if first > last or (last - first).days >= 31:
        raise ValueError("Use a date window of 1–31 calendar days")
    return {"version": 1, "field": "DateAddedTimestamp", "start": first.isoformat(),
            "end": last.isoformat(), "upper_boundary": (last + dt.timedelta(days=1)).isoformat(),
            "includes_next_midnight": True, "scope": "dated_metadata_supplement",
            "withdrawals_detected": False, "older_revisions_detected": False}


class WindowClient:
    def __init__(self, client, specification, directory, cancelled=None):
        self.client, self.specification, self.directory = client, specification, directory
        self.cancelled = cancelled

    def search(self, *, library="main", filters=None, skip=0):
        if self.cancelled and self.cancelled():
            raise InterruptedError("Recent-record collection stopped")
        require_space(self.directory)
        spec = self.specification
        date_filter = {"field": "DateAddedTimestamp", "value":
                       f"(DateAddedTimestamp ge '{spec['start']}') and (DateAddedTimestamp le '{spec['upper_boundary']}')"}
        response = self.client.search(library=library, filters=[date_filter, *(filters or [])], skip=skip)
        lower = dt.datetime.fromisoformat(spec["start"])
        upper = dt.datetime.fromisoformat(spec["upper_boundary"])
        for document in unpack_search(response):
            try:
                stamp = dt.datetime.fromisoformat(document["DateAddedTimestamp"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("A recent-record response has no usable DateAddedTimestamp") from None
            # NRC's documented field format is a calendar timestamp without an
            # offset. Do not silently interpret a changed format or ignored filter.
            if stamp.tzinfo is not None or not lower <= stamp <= upper:
                raise ValueError("API returned a record outside the requested calendar window")
            if library not in {"main", "legacy"} or is_legacy(document) != (library == "legacy"):
                raise ValueError("API returned a record from the wrong public library")
        require_space(self.directory, extra_bytes=max(32 * 1024 * 1024, len(json.dumps(response).encode()) * 12))
        if self.cancelled and self.cancelled():
            raise InterruptedError("Recent-record collection stopped before committing a page")
        return response

    search_count = search


def collect(directory, client, start, end, *, max_leaf=1000, cancelled=None):
    specification = window(start, end)
    if type(max_leaf) is not int or not 1 <= max_leaf <= 1000:
        raise ValueError("Use a leaf allowance between 1 and 1,000 records")
    directory = Path(directory).resolve()
    ancestor = directory
    while not ancestor.exists():
        ancestor = ancestor.parent
    require_space(ancestor)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A collector already owns this supplement directory") from None
        manifest = directory / "window.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != specification:
                raise ValueError("This supplement belongs to a different date window; use another directory")
        else:
            if (directory / "catalog.sqlite").exists():
                raise ValueError("Refusing to treat an existing catalog as a dated supplement")
            atomic_write(manifest, json.dumps(specification, indent=2) + "\n")
        catalog = Catalog(directory)
        bounded = WindowClient(client, specification, directory, cancelled)
        state, detail = "running", None
        try:
            try:
                while True:
                    require_space(directory)
                    if cancelled and cancelled():
                        raise InterruptedError("Recent-record collection stopped")
                    if not catalog.step(bounded, max_leaf):
                        if catalog.reconcile(bounded):
                            state = "window_count_matched"
                            break
                        if not catalog.db.execute("SELECT 1 FROM audit_queue LIMIT 1").fetchone():
                            state, detail = "needs_review", "A partition needs review before this window can be reconciled"
                            break
            except StoragePause as error:
                state, detail = "storage_pause", str(error)
            except BudgetStop as error:
                state, detail = "budget_pause", str(error)
            except TemporaryRequestError:
                state, detail = "network_wait", "Temporary API failure; committed pages retained"
            except InterruptedError as error:
                state, detail = "stopped", str(error)
            except (ValueError, RuntimeError) as error:
                state, detail = "needs_review", str(error)
            result = {"state": state, "detail": detail, "window": specification, **catalog.summary(),
                      "full_archive_verified": False, "full_text_acquired": False,
                      "base_catalog_modified": False}
            atomic_write(directory / "STATUS.json", json.dumps(result, indent=2) + "\n")
            return result
        finally:
            catalog.db.close()


def main(argv=None):
    from .client import AdamsClient
    from .config import load_api_key
    from .store import Store
    from .transport import PersistentOpener
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--rate-db", default="data/adams.sqlite")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-leaf", type=int, default=1000)
    args = parser.parse_args(argv)
    window(args.start, args.end)
    if not 1 <= args.requests <= 1000:
        parser.error("Use 1–1,000 requests per run")
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    def cancelled():
        return stopping[0] or (Path(args.directory) / "STOP").exists()
    rate, transport = Store(args.rate_db), PersistentOpener()
    try:
        client = AdamsClient(rate, load_api_key(), max_requests=args.requests, rps=2, daily_limit=90000,
                             opener=transport, include_search_content=False, cancelled=cancelled)
        result = collect(args.directory, client, args.start, args.end, max_leaf=args.max_leaf, cancelled=cancelled)
        print(json.dumps(result, indent=2))
    finally:
        transport.close()
        rate.db.close()


if __name__ == "__main__":
    main()
