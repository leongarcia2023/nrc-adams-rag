import datetime as dt
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

from adams_rag.client import BudgetStop
from adams_rag.recent import WindowClient, collect, window
from adams_rag.storage import StoragePause


def document(accession, legacy=False, stamp="2026-10-30 12:30"):
    return {"AccessionNumber": accession, "IdT": accession, "DocumentTitle": "Recent report",
            "IsLegacy": "Yes" if legacy else "No", "DateAddedTimestamp": stamp, "content": "Search excerpt"}


class Server:
    def __init__(self, remaining=100):
        self.docs = {"main": [document(f"AA{i}") for i in range(5)],
                     "legacy": [document("7902060261", legacy=True)]}
        self.calls, self.remaining = [], remaining

    def search(self, library, filters, skip):
        if self.remaining <= 0:
            raise BudgetStop("Per-run budget")
        self.remaining -= 1
        self.calls.append((library, skip, filters))
        docs = self.docs[library]
        for f in filters:
            if f["field"] == "DateAddedTimestamp":
                lower, upper = re.findall(r"'([^']+)'", f["value"])
                docs = [d for d in docs if dt.datetime.fromisoformat(lower) <= dt.datetime.fromisoformat(d["DateAddedTimestamp"]) <= dt.datetime.fromisoformat(upper)]
            else:
                value = f["value"].casefold()
                def accept(d):
                    a = d["AccessionNumber"].casefold()
                    return {"starts": a.startswith(value), "notstarts": not a.startswith(value),
                            "equals": a == value, "notequals": a != value}[f["operator"]]
                docs = [d for d in docs if accept(d)]
        docs = sorted(docs, key=lambda d: d["AccessionNumber"])
        return {"count": len(docs), "results": [{"document": d} for d in docs[skip:skip+2]]}


class RecentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.directory = self.root / "recent"

    def tearDown(self):
        self.tmp.cleanup()

    def collect(self, server, **kwargs):
        return collect(self.directory, server, "2026-10-29", "2026-10-31", **kwargs)

    def test_dated_supplement_reconciles_both_libraries_and_never_indexes_excerpts(self):
        server = Server()
        result = self.collect(server, max_leaf=2)
        self.assertEqual((result["state"], result["catalog_records"]), ("window_count_matched", 6))
        self.assertTrue(result["enumeration_count_matched"])
        self.assertFalse(result["full_archive_verified"])
        self.assertFalse(result["base_catalog_modified"])
        self.assertFalse(result["full_text_acquired"])
        self.assertTrue(all(sum(f["field"] == "DateAddedTimestamp" for f in filters) == 1 for _, _, filters in server.calls))
        db = sqlite3.connect(self.directory / "catalog.sqlite")
        for row in db.execute("SELECT metadata FROM items"):
            self.assertNotIn("content", json.loads(zlib.decompress(row[0])))
        db.close()

    def test_request_budget_resumes_committed_pages(self):
        first = self.collect(Server(remaining=2))
        self.assertEqual((first["state"], first["catalog_records"]), ("budget_pause", 3))
        server = Server()
        second = self.collect(server)
        self.assertEqual((second["state"], second["catalog_records"]), ("window_count_matched", 6))
        self.assertEqual(server.calls[0][:2], ("main", 2))

    def test_completed_window_is_an_immutable_observation_and_new_directory_rechecks(self):
        server = Server()
        self.collect(server)
        calls = len(server.calls)
        server.docs["main"].append(document("AZ1"))
        self.assertEqual(self.collect(server)["catalog_records"], 6)
        self.assertEqual(len(server.calls), calls)
        self.assertEqual(collect(self.root / "recheck", server, "2026-10-29", "2026-10-31")["catalog_records"], 7)

    def test_wrong_window_or_existing_base_catalog_is_not_adopted(self):
        self.collect(Server())
        with self.assertRaisesRegex(ValueError, "different date window"):
            collect(self.directory, Server(), "2026-10-28", "2026-10-31")
        base = self.root / "base"; base.mkdir()
        source = base / "catalog.sqlite"; source.write_bytes(b"preserved base")
        with self.assertRaisesRegex(ValueError, "existing catalog"):
            collect(base, Server(), "2026-10-29", "2026-10-31")
        self.assertEqual(source.read_bytes(), b"preserved base")

    def test_storage_drop_after_response_never_commits_that_page(self):
        with patch("adams_rag.recent.require_space", side_effect=[None, None, None, StoragePause("low")]):
            result = self.collect(Server())
        self.assertEqual((result["state"], result["catalog_records"]), ("storage_pause", 0))
        db = sqlite3.connect(self.directory / "catalog.sqlite")
        self.assertEqual(db.execute("SELECT sum(offset) FROM partitions").fetchone()[0], 0)
        db.close()

    def test_ignored_date_filter_missing_timestamp_and_wrong_library_are_rejected(self):
        for record in [document("AA1", stamp="2020-01-01"), document("AA1", legacy=True),
                       {"AccessionNumber": "AA1"}, document("AA1", stamp="2026-10-30T12:00:00+00:00")]:
            class Invalid:
                def search(self, **kwargs):
                    return {"count": 1, "results": [{"document": record}]}
            with self.subTest(record=record), self.assertRaises(ValueError):
                WindowClient(Invalid(), window("2026-10-29", "2026-10-31"), self.root).search()

    def test_next_midnight_boundary_is_explicit_and_later_records_excluded(self):
        server = Server()
        server.docs = {"main": [document("AA1", stamp="2026-11-01 00:00"), document("AA2", stamp="2026-11-01 00:01")], "legacy": []}
        result = self.collect(server)
        self.assertEqual(result["catalog_records"], 1)
        self.assertTrue(result["window"]["includes_next_midnight"])

    def test_cancellation_and_lock_exclude_parallel_mutation(self):
        import fcntl
        result = self.collect(Server(), cancelled=lambda: True)
        self.assertEqual((result["state"], result["catalog_records"]), ("stopped", 0))
        with (self.directory / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                self.collect(Server())

    def test_invalid_window_is_rejected_before_directory_creation(self):
        for start, end in [("bad", "2026-10-31"), ("2026-11-01", "2026-10-31"), ("2026-09-01", "2026-10-31")]:
            with self.assertRaises(ValueError):
                collect(self.directory, Server(), start, end)
        self.assertFalse(self.directory.exists())


if __name__ == "__main__":
    unittest.main()
