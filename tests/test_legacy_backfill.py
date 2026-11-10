import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

from adams_rag.backfill import backfill
from adams_rag.client import AdamsHTTPError, BudgetStop
from adams_rag.research import research
from adams_rag.store import Store
from adams_rag.text_quality import UNAVAILABLE_NOTICE, get


def document(accession="9803170214", identity="FIRST", content="Valve fuse failure caused a manual trip.", **extra):
    return {"AccessionNumber": accession, "IdT": identity, "IsLegacy": "Yes", "content": content, **extra}


class LegacyBackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = Store(root / "research.sqlite")
        self.catalog = root / "catalog.sqlite"
        db = sqlite3.connect(self.catalog)
        db.executescript("""CREATE TABLE libraries(library,final_count,finished,state);
            INSERT INTO libraries VALUES('legacy',3,'2026-10-31','complete');
            CREATE TABLE items(library,accession,is_package,metadata);""")
        for accession, identity in [("9803170214", "FIRST"), ("9803170214", "SECOND"), ("7809190190", "THIRD")]:
            doc = document(accession, identity, "", DocumentTitle="Historical valve report")
            db.execute("INSERT INTO items VALUES('legacy',?,0,?)", (accession, zlib.compress(json.dumps(doc).encode())))
        db.commit(); db.close()

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def run_batch(self, client, **kwargs):
        return backfill(self.store, client, self.catalog, library="legacy", **kwargs)

    def test_shared_accession_never_assigns_one_records_text_to_its_sibling(self):
        class Client:
            def request(self, accession):
                return {"document": document() if accession == "9803170214" else document(accession, "THIRD", "")}
        result = self.run_batch(Client())
        self.assertEqual((result["indexed"], result["legacy_identity_mismatch"], result["legacy_metadata"]), (1, 1, 1))
        self.assertEqual(result["state"], "catalog_exhausted")
        self.assertEqual([r[0] for r in self.store.db.execute("SELECT accession FROM documents WHERE status='indexed'")], ["nrc:legacy:id:first"])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM documents WHERE accession='nrc:legacy:id:second'").fetchone()[0], 0)
        gap = self.store.db.execute("SELECT record_id,detail FROM backfill_records WHERE outcome='legacy_identity_mismatch'").fetchone()
        self.assertEqual(gap["record_id"], "id:second")
        self.assertEqual(json.loads(gap["detail"])["returned_record_id"], "id:first")
        quality = get(self.store.db, "nrc:legacy:id:first")
        self.assertEqual(quality["scope"], "legacy_indexed_text")
        self.assertIn("legacy_content_scope_unverified", quality["flags"])
        result = research(self.store, "valve fuse", live=False, generate=False)
        self.assertEqual(result["sources"][0]["source_record_id"], "id:first")
        self.assertEqual(result["sources"][0]["source_version"]["state"], "matched_retained_text")
        self.assertTrue(any("abstract or a full document" in w for w in result["warnings"]))

    def test_budget_pause_and_restart_preserve_exact_record_progress(self):
        class Client:
            def request(self, accession):
                return {"document": document()}
        first = self.run_batch(Client(), max_documents=1)
        class Pause:
            def request(self, accession):
                raise BudgetStop("pause")
        paused = self.run_batch(Pause())
        self.assertEqual((paused["state"], paused["last_catalog_row"]), ("budget_pause", 1))
        resumed = self.run_batch(Client(), max_documents=1)
        self.assertEqual((resumed["last_catalog_row"], resumed["legacy_identity_mismatch"]), (2, 1))
        self.assertEqual(first["job"], resumed["job"])

    def test_prior_metadata_only_cursor_is_not_reused_for_text_acquisition(self):
        initial = self.run_batch(object(), cancelled=lambda: True)
        old_job = hashlib.sha256(json.dumps([str(self.catalog.resolve()), {
            "library": "legacy", "final_count": 3, "finished": "2026-10-31", "state": "complete"}], sort_keys=True).encode()).hexdigest()
        with self.store.db:
            self.store.db.execute("INSERT INTO backfill_cursors(job,library,catalog,last_row,state) VALUES(?,?,?,?,?)",
                                  (old_job, "legacy", str(self.catalog), 3, "catalog_exhausted"))
        class Client:
            def request(self, accession):
                return {"document": document()}
        result = self.run_batch(Client(), max_documents=1)
        self.assertNotEqual(result["job"], old_job)
        self.assertEqual((result["job"], result["indexed"], result["last_catalog_row"]), (initial["job"], 1, 1))

    def test_missing_or_wrong_library_and_wrong_accession_do_not_advance(self):
        for doc in [document(IdT=None), document(IsLegacy="No"), document(accession="9803170999")]:
            class Client:
                def request(self, accession):
                    return {"document": doc}
            with self.assertRaises(ValueError):
                self.run_batch(Client(), max_documents=1)
            self.assertEqual(self.store.db.execute("SELECT last_row FROM backfill_cursors").fetchone()[0], 0)
            self.assertEqual(self.store.status()["chunks"], 0)

    def test_unavailable_notice_is_metadata_only_even_with_matching_identity(self):
        class Client:
            def request(self, accession):
                return {"document": document(content=UNAVAILABLE_NOTICE)}
        result = self.run_batch(Client(), max_documents=1)
        self.assertEqual((result["legacy_metadata"], result["indexed"]), (1, 0))
        self.assertEqual(self.store.status()["chunks"], 0)
        self.assertEqual(self.store.status()["document_versions"], 0)

    def test_already_indexed_skip_uses_record_identity_not_public_accession(self):
        with self.store.db:
            self.store.put(document())
        calls = []
        class Client:
            def request(self, accession):
                calls.append(accession)
                return {"document": document(identity="SECOND")}
        result = self.run_batch(Client(), max_documents=1)
        self.assertEqual((result["already_indexed"], result["indexed"], result["last_catalog_row"]), (1, 1, 2))
        self.assertEqual(calls, ["9803170214"])
        self.assertEqual(self.store.status()["documents"], {"indexed": 2})

    def test_http_errors_are_durable_only_for_unavailable_records(self):
        class Client:
            def request(self, accession):
                raise AdamsHTTPError(401)
        with self.assertRaises(AdamsHTTPError):
            self.run_batch(Client())
        self.assertEqual(self.store.db.execute("SELECT last_row FROM backfill_cursors").fetchone()[0], 0)
        class Missing:
            def request(self, accession):
                raise AdamsHTTPError(404)
        result = self.run_batch(Missing(), max_documents=1)
        self.assertEqual((result["unavailable"], result["last_catalog_row"]), (1, 1))

    def test_storage_drop_does_not_publish_a_mismatch_or_advance(self):
        class Client:
            def request(self, accession):
                return {"document": document(identity="SECOND")}
        from types import SimpleNamespace
        with patch("shutil.disk_usage", side_effect=[SimpleNamespace(free=n) for n in (3_000_000_000, 3_000_000_000, 510_000_000)]):
            result = self.run_batch(Client(), min_free_gb=.5)
        self.assertEqual((result["state"], result["last_catalog_row"]), ("storage_pause", 0))
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM backfill_records").fetchone()[0], 0)
        self.assertFalse((self.store.path.parent / "raw").exists())


if __name__ == "__main__":
    unittest.main()
