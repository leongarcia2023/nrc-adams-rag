import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

from adams_rag.backfill import backfill
from adams_rag.client import AdamsHTTPError, BudgetStop
from adams_rag.store import Store


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = Store(root / "research.sqlite")
        self.catalog = root / "catalog.sqlite"
        source = sqlite3.connect(self.catalog)
        source.executescript("""CREATE TABLE libraries(library,final_count,finished,state);
          INSERT INTO libraries VALUES('main',3,'2026-10-29','complete');
          CREATE TABLE items(library,accession,is_package,metadata);""")
        for accession in ["ML24017A120", "ML24017A121", "ML24017A122"]:
            source.execute("INSERT INTO items VALUES('main',?,0,?)", (accession, zlib.compress(json.dumps(
                {"AccessionNumber": accession, "DocumentTitle": "NRC report", "DocketNumber": ["05000373"]}).encode())))
        source.commit(); source.close()

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_resume_skips_indexed_documents_and_merges_catalog_metadata(self):
        calls = []
        class Client:
            def request(self, accession):
                calls.append(accession)
                return {"document": {"AccessionNumber": accession, "content": "Evidence about the valve."}}
        a = backfill(self.store, Client(), self.catalog, max_documents=1)
        b = backfill(self.store, Client(), self.catalog, max_documents=2)
        c = backfill(self.store, Client(), self.catalog, max_documents=2)
        self.assertEqual((a["indexed"], b["indexed"], c["state"]), (1, 2, "catalog_exhausted"))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(self.store.lexical("valve", docket="05000373")), 3)

    def test_budget_pause_does_not_advance_past_unfetched_record(self):
        class Client:
            def request(self, accession):
                raise BudgetStop("Local request budget reached")
        result = backfill(self.store, Client(), self.catalog)
        self.assertEqual(result["state"], "budget_pause")
        self.assertEqual(result["last_catalog_row"], 0)

    def test_wrong_response_keeps_cursor_and_no_false_indexed_record(self):
        class Client:
            def request(self, accession):
                return {"document": {"AccessionNumber": "ML24017A999", "content": "Wrong record"}}
        with self.assertRaises(ValueError):
            backfill(self.store, Client(), self.catalog)
        self.assertEqual(self.store.db.execute("SELECT last_row FROM backfill_cursors").fetchone()[0], 0)
        self.assertEqual(self.store.status()["chunks"], 0)

    def test_low_disk_space_stops_without_network_request(self):
        class Client:
            def request(self, accession):
                raise AssertionError("Network must not be used below the storage reserve")
        with patch("shutil.disk_usage") as disk:
            disk.return_value.free = 400_000_000
            result = backfill(self.store, Client(), self.catalog)
        self.assertEqual(result["state"], "storage_pause")

    def test_deleted_record_is_a_durable_gap_and_does_not_block_next_record(self):
        class Client:
            def request(self, accession):
                if accession == "ML24017A120":
                    raise AdamsHTTPError(404)
                return {"document": {"AccessionNumber": accession, "content": "Retained evidence"}}
        result = backfill(self.store, Client(), self.catalog, max_documents=2)
        self.assertEqual((result["unavailable"], result["indexed"], result["last_catalog_row"]), (1, 1, 2))
        self.assertEqual(dict(self.store.db.execute("SELECT outcome,count FROM backfill_totals").fetchall()),
                         {"unavailable": 1, "indexed": 1})

    def test_authentication_failure_never_skips_a_catalog_record(self):
        class Client:
            def request(self, accession):
                raise AdamsHTTPError(401)
        with self.assertRaises(AdamsHTTPError):
            backfill(self.store, Client(), self.catalog)
        self.assertEqual(self.store.db.execute("SELECT last_row FROM backfill_cursors").fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM backfill_records").fetchone()[0], 0)

    def test_free_space_drop_after_response_preserves_cursor_and_does_not_write_text(self):
        class Client:
            def request(self, accession):
                return {"document": {"AccessionNumber": accession, "content": "Evidence"}}
        with patch("shutil.disk_usage") as disk:
            from types import SimpleNamespace
            disk.side_effect = [SimpleNamespace(free=3_000_000_000), SimpleNamespace(free=3_000_000_000), SimpleNamespace(free=510_000_000)]
            result = backfill(self.store, Client(), self.catalog, min_free_gb=.5)
        self.assertEqual((result["state"], result["last_catalog_row"]), ("storage_pause", 0))
        self.assertEqual(self.store.status()["chunks"], 0)
        self.assertFalse((self.store.path.parent / "raw").exists())

    def test_cancellation_keeps_unattempted_record_available(self):
        result = backfill(self.store, object(), self.catalog, cancelled=lambda: True)
        self.assertEqual((result["state"], result["last_catalog_row"]), ("stopped", 0))

    def test_only_one_writer_can_advance_the_cursor(self):
        import fcntl
        with self.store.path.with_suffix(".backfill.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                backfill(self.store, object(), self.catalog)

    def test_unsupported_legacy_accessions_remain_auditable_without_requests(self):
        c = sqlite3.connect(self.catalog)
        c.execute("UPDATE libraries SET library='legacy'")
        c.execute("UPDATE items SET library='legacy'")
        for rowid, accession in c.execute("SELECT rowid,accession FROM items").fetchall():
            c.execute("UPDATE items SET metadata=? WHERE rowid=?", (zlib.compress(json.dumps({
                "AccessionNumber": accession, "IdT": f"RECORD-{rowid}", "IsLegacy": "Yes"}).encode()), rowid))
        c.commit(); c.close()
        class Client:
            def request(self, **kwargs):
                raise AssertionError("Unsupported legacy accessions must not be fetched")
        result = backfill(self.store, Client(), self.catalog, library="legacy")
        self.assertEqual(result["unsupported_accessions"], 3)
        self.assertEqual(result["attempted"], 0)
        self.assertEqual({r[0] for r in self.store.db.execute("SELECT record_id FROM backfill_records")},
                         {"id:record-1", "id:record-2", "id:record-3"})
        self.assertFalse(result["full_archive_verified"])

    def test_legacy_response_cannot_be_mislabelled_as_a_main_document(self):
        class Client:
            def request(self, accession):
                return {"document": {"AccessionNumber": accession, "IdT": "OTHER", "IsLegacy": "Yes", "content": "Legacy text"}}
        with self.assertRaisesRegex(ValueError, "legacy record"):
            backfill(self.store, Client(), self.catalog)
        self.assertEqual(self.store.db.execute("SELECT last_row FROM backfill_cursors").fetchone()[0], 0)
        self.assertEqual(self.store.status()["chunks"], 0)

    def test_older_audit_identity_is_repaired_without_advancing_a_paused_cursor(self):
        class Client:
            def request(self, accession):
                return {"document": {"AccessionNumber": accession, "content": "Evidence"}}
        first = backfill(self.store, Client(), self.catalog, max_documents=1)
        with self.store.db:
            self.store.db.execute("UPDATE backfill_records SET record_id=NULL")
        class Paused:
            def request(self, accession):
                raise BudgetStop("test pause")
        second = backfill(self.store, Paused(), self.catalog, max_documents=1)
        self.assertEqual(first["last_catalog_row"], second["last_catalog_row"])
        self.assertEqual(self.store.db.execute("SELECT record_id FROM backfill_records").fetchone()[0],
                         "accession:ml24017a120")
