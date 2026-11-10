import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

from adams_rag.catalog_collection import discover, lookup, lookup_available


class CollectionLookupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.catalog("base", [
            ("main", "id:a", "ML00001A001", "Original"),
            ("legacy", "id:first", "9901010001", "First"),
            ("legacy", "id:second", "9901010001", "Second")])
        self.new = self.catalog("new", [
            ("main", "id:a", "ML00001A002", "Reobserved"),
            ("main", "id:new", "ML00001A003", "New"),
            ("legacy", "id:first", "9901010001", "Updated first")])

    def tearDown(self):
        self.tmp.cleanup()

    def catalog(self, name, records):
        folder = self.root / name
        folder.mkdir()
        path = folder / "catalog.sqlite"
        db = sqlite3.connect(path)
        db.executescript("""CREATE TABLE libraries(library,final_count,finished,state);
            INSERT INTO libraries VALUES('main',1,'2026-10-29','complete');
            INSERT INTO libraries VALUES('legacy',2,'2026-10-29','complete');
            CREATE TABLE items(library,record_id,accession,is_package,metadata,PRIMARY KEY(library,record_id));
            CREATE INDEX item_accession_lookup_v1 ON items(accession COLLATE NOCASE,library);""")
        for library, identity, accession, title in records:
            db.execute("INSERT INTO items VALUES(?,?,?,0,?)", (library, identity, accession,
                zlib.compress(json.dumps({"DocumentTitle": title, "content": "not full text",
                                          "SearchExcerpt": "not evidence"}).encode())))
        db.commit(); db.close()
        return path

    def test_new_records_and_changes_preserve_identity_and_provenance(self):
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.base, self.new)]
        self.assertEqual(lookup(self.base, [self.new], accession="ML00001A001")["matched_records"], 0)
        result = lookup(self.base, [self.new], record_id="{A}")
        record = result["records"][0]
        self.assertEqual(record["accession"], "ML00001A002")
        self.assertEqual(record["metadata"], {"DocumentTitle": "Reobserved"})
        self.assertEqual(len(record["observations"]), 2)
        self.assertEqual(record["selected_snapshot"], result["catalog_snapshots"][-1]["snapshot_id"])
        self.assertFalse(result["withdrawals_detected"])
        self.assertFalse(result["current_authority_verified"])
        self.assertEqual(lookup(self.base, [self.new], accession="ml00001a003")["matched_records"], 1)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.base, self.new)])

    def test_duplicate_accessions_remain_distinct_and_paginate_after_deduplication(self):
        first = lookup(self.base, [self.new], accession="9901010001", limit=1)
        self.assertEqual(first["matched_records"], 2)
        self.assertEqual(first["records"][0]["metadata"]["DocumentTitle"], "Updated first")
        second = lookup(self.base, [self.new], accession="9901010001", limit=1, cursor=first["next_cursor"])
        self.assertEqual(second["records"][0]["record_id"], "id:second")
        self.assertIsNone(second["next_cursor"])
        for kwargs in [{"accession": "ML00001A003"}, {"accession": "9901010001", "library": "legacy"}]:
            with self.assertRaises(ValueError):
                lookup(self.base, [self.new], cursor=first["next_cursor"], **kwargs)
        with self.assertRaises(ValueError):
            lookup(self.base, accession="9901010001", cursor=first["next_cursor"])
        db = sqlite3.connect(self.new)
        db.execute("UPDATE libraries SET finished='2026-11-01'"); db.commit(); db.close()
        with self.assertRaises(ValueError):
            lookup(self.base, [self.new], accession="9901010001", cursor=first["next_cursor"])

    def test_same_record_id_across_libraries_stays_separate(self):
        another = self.catalog("another", [("main", "id:first", "ML00001A004", "Different library")])
        result = lookup(self.base, [another], record_id="FIRST")
        self.assertEqual({r["library"] for r in result["records"]}, {"main", "legacy"})
        self.assertEqual(lookup(self.base, [another], record_id="FIRST", library="main")["matched_records"], 1)

    def test_missing_index_and_unfinished_snapshots_fail_closed(self):
        db = sqlite3.connect(self.new)
        db.execute("DROP INDEX item_accession_lookup_v1"); db.commit(); db.close()
        with self.assertRaisesRegex(RuntimeError, "index is not ready"):
            lookup(self.base, [self.new], accession="ML00001A003")
        self.assertEqual(lookup(self.base, [self.new], record_id="new")["matched_records"], 1)
        db = sqlite3.connect(self.new)
        db.execute("UPDATE libraries SET finished=NULL"); db.commit(); db.close()
        with self.assertRaisesRegex(RuntimeError, "reconciled"):
            lookup(self.base, [self.new], record_id="new")

    def test_bounds_and_invalid_values_never_return_partial_counts(self):
        with patch("adams_rag.catalog_collection.MAX_MATCHES", 1):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                lookup(self.base, [self.new], accession="9901010001")
        for kwargs in [{}, {"accession": ""}, {"record_id": "a", "limit": True},
                       {"record_id": "a", "cursor": "broken"}]:
            with self.assertRaises(ValueError):
                lookup(self.base, [self.new], **kwargs)
        with self.assertRaises(ValueError):
            lookup(self.base, [self.base], record_id="a")
        self.assertEqual(lookup(self.base, [self.new], accession="' OR 1=1 --")["matched_records"], 0)

    def test_discovery_orders_observations_and_reports_pending_supplements(self):
        for path, observed in [(self.base, "2026-11-02T00:00:00+00:00"), (self.new, "2026-11-01T00:00:00+00:00")]:
            window = {"scope": "dated_metadata_supplement"}
            (path.parent / "window.json").write_text(json.dumps(window))
            (path.parent / "STATUS.json").write_text(json.dumps({"window": window,
                "state": "window_count_matched", "enumeration_count_matched": True,
                "libraries": [{"finished": observed}], "updated_utc": "2026-11-10T00:00:00+00:00"}))
        pending = self.root / "unfinished"
        pending.mkdir(); (pending / "window.json").write_text("{}")
        sources, excluded = discover(self.root)
        self.assertEqual(sources, [self.new, self.base])
        self.assertEqual(excluded, ["unfinished"])
        # Use a separate directory containing only pending observations for API behavior.
        (self.base.parent / "window.json").unlink()
        (self.new.parent / "window.json").unlink()
        result = lookup_available(self.base, self.root, accession="9901010001")
        self.assertEqual(result["pending_supplements"], ["unfinished"])
        self.assertTrue(result["warnings"])

    def test_malformed_supplement_manifest_is_reported_without_hiding_base_results(self):
        (self.new.parent / "window.json").write_text("[]")
        (self.new.parent / "STATUS.json").write_text("[]")
        result = lookup_available(self.base, self.root, accession="9901010001")
        self.assertEqual(result["matched_records"], 2)
        self.assertEqual(result["pending_supplements"], ["new"])


if __name__ == "__main__":
    unittest.main()
