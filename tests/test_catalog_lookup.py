import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

from adams_rag.catalog_lookup import build_index, lookup
from adams_rag.storage import StoragePause


class CatalogLookupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "catalog.sqlite"
        c = sqlite3.connect(self.path)
        c.executescript("""CREATE TABLE libraries(library,final_count,finished,state);
            INSERT INTO libraries VALUES('main',1,'2026-10-29','complete');
            INSERT INTO libraries VALUES('legacy',2,'2026-10-29','complete');
            CREATE TABLE items(library,record_id,accession,is_package,metadata,PRIMARY KEY(library,record_id));""")
        for library, identity, accession, title in [
            ("main", "id:main", "ML24017A120", "Current report"),
            ("legacy", "id:first", "9901010001", "First legacy record"),
            ("legacy", "id:second", "9901010001", "Separate legacy record")]:
            c.execute("INSERT INTO items VALUES(?,?,?,0,?)", (library, identity, accession,
                zlib.compress(json.dumps({"AccessionNumber": accession, "DocumentTitle": title,
                                          "content": "Untrusted search excerpt"}).encode())))
        c.commit(); c.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_accessions_keep_separate_record_ids_and_metadata_scope(self):
        build_index(self.path)
        result = lookup(self.path, "9901010001")
        self.assertEqual(result["matched_records"], 2)
        self.assertEqual(len({r["source_key"] for r in result["records"]}), 2)
        self.assertTrue(all(not r["full_text_included"] and "content" not in r["metadata"] for r in result["records"]))
        self.assertFalse(result["full_archive_verified"])

    def test_pagination_is_bound_to_query_and_snapshot(self):
        build_index(self.path)
        first = lookup(self.path, "9901010001", limit=1)
        second = lookup(self.path, "9901010001", limit=1, cursor=first["next_cursor"])
        self.assertEqual(second["records"][0]["record_id"], "id:second")
        self.assertIsNone(second["next_cursor"])
        with self.assertRaises(ValueError):
            lookup(self.path, "ML24017A120", cursor=first["next_cursor"])
        c = sqlite3.connect(self.path)
        c.execute("UPDATE libraries SET finished='2026-10-31'"); c.commit(); c.close()
        with self.assertRaises(ValueError):
            lookup(self.path, "9901010001", cursor=first["next_cursor"])

    def test_case_insensitive_accession_and_record_id_lookup(self):
        build_index(self.path)
        self.assertEqual(lookup(self.path, "ml24017a120")["matched_records"], 1)
        self.assertEqual(lookup(self.path, record_id="{FIRST}", library="legacy")["records"][0]["record_id"], "id:first")
        self.assertEqual(lookup(self.path, "9901010001", library="main")["matched_records"], 0)

    def test_missing_index_does_not_fall_back_to_an_unbounded_scan(self):
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            lookup(self.path, "ML24017A120")
        self.assertEqual(lookup(self.path, record_id="id:first", library="legacy")["matched_records"], 1)

    def test_low_storage_and_cancel_leave_catalog_unchanged(self):
        with patch("shutil.disk_usage") as disk:
            disk.return_value.free = 400_000_000
            with self.assertRaises(StoragePause):
                build_index(self.path)
        with self.assertRaises(InterruptedError):
            build_index(self.path, cancelled=lambda: True)
        c = sqlite3.connect(self.path)
        self.assertEqual(c.execute("SELECT count(*) FROM items").fetchone()[0], 3)
        self.assertEqual(c.execute("SELECT count(*) FROM sqlite_master WHERE type='index' AND name='item_accession_lookup_v1'").fetchone()[0], 0)
        c.close()

    def test_invalid_and_injected_values_cannot_broaden_lookup(self):
        build_index(self.path)
        self.assertEqual(lookup(self.path, "' OR 1=1 --")["matched_records"], 0)
        for kwargs in [{}, {"accession": ""}, {"accession": "A", "limit": 100000},
                       {"accession": "A", "library": "private"}, {"accession": "A", "cursor": "broken"}]:
            with self.assertRaises(ValueError):
                lookup(self.path, **kwargs)
        self.assertFalse(build_index(self.path)["created"])

    def test_storage_drop_during_index_build_rolls_back_only_the_new_index(self):
        from types import SimpleNamespace
        c = sqlite3.connect(self.path)
        c.executemany("INSERT INTO items VALUES('main',?,?,0,?)", (
            (f"id:bulk-{n}", f"ML{n:09}", zlib.compress(b'{}')) for n in range(3000)))
        c.execute("UPDATE libraries SET final_count=3001 WHERE library='main'"); c.commit(); c.close()
        readings = iter([3_000_000_000, 3_000_000_000])
        with patch("shutil.disk_usage", side_effect=lambda _: SimpleNamespace(free=next(readings, 400_000_000))):
            with self.assertRaises(StoragePause):
                build_index(self.path)
        c = sqlite3.connect(self.path)
        self.assertEqual(c.execute("SELECT count(*) FROM items").fetchone()[0], 3003)
        self.assertEqual(c.execute("SELECT count(*) FROM sqlite_master WHERE name='item_accession_lookup_v1'").fetchone()[0], 0)
        c.close()


if __name__ == "__main__":
    unittest.main()
