import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

from adams_rag.catalog_lookup import build_index
from adams_rag.legacy_links import build, default_path, lookup, references, snapshot, upgrade
from adams_rag.storage import StoragePause
from adams_rag.research import research
from adams_rag.store import Store


class LegacyLinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.catalog = Path(self.tmp.name) / "catalog.sqlite"
        db = sqlite3.connect(self.catalog)
        db.executescript("""CREATE TABLE libraries(library,final_count,finished,state);
            INSERT INTO libraries VALUES('main',4,'2026-10-29','complete');
            INSERT INTO libraries VALUES('legacy',2,'2026-10-29','complete');
            CREATE TABLE items(library,record_id,accession,is_package,metadata,PRIMARY KEY(library,record_id));
            CREATE TABLE inventory_metrics(records); INSERT INTO inventory_metrics VALUES(6);""")
        records = [
            ("main", "id:one", "ML20001A001", 0, {"DocumentReportNumber": ["NUDOCS 7902060261", "NUDOCS 7902060261"], "DocumentTitle": "Digitized report"}),
            ("legacy", "id:old-one", "7902060261", 0, {}),
            ("main", "id:two", "ML20001A002", 0, {"DocumentReportNumber": ["nudocs 7902060261", "NUDOCS 7902060262"]}),
            ("legacy", "id:old-two", "7902060261", 0, {}),
            ("main", "id:package", "ML20001A003", 1, {"DocumentReportNumber": ["NUDOCS 7902060261"]}),
            ("main", "id:unrelated", "ML20001A004", 0, {"DocumentTitle": "NUDOCS 7902060261", "DocumentReportNumber": ["NUDOCS PKG 7902060261"]}),
        ]
        for library, identity, accession, package, metadata in records:
            db.execute("INSERT INTO items VALUES(?,?,?,?,?)", (library, identity, accession, package,
                zlib.compress(json.dumps({"AccessionNumber": accession, **metadata}).encode())))
        db.commit(); db.close()
        build_index(self.catalog)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parser_requires_explicit_whole_report_reference(self):
        found, unparsed = references({"DocumentReportNumber": ["NUDOCS 0000000001", "nudocs 0000000001", "NUDOCS PKG 0000000002", "See NUDOCS 0000000003", "NUDOCS 00000000045", "unrelated 0000000005"], "Comment": "NUDOCS 0000000006"})
        self.assertEqual(set(found), {"0000000001"})
        self.assertEqual(unparsed, 3)
        self.assertEqual(references({"DocumentReportNumber": "NUDOCS 0000000001"})[0], {"0000000001": "NUDOCS 0000000001"})

    def test_comma_separated_report_numbers_keep_explicit_reference_and_original_field(self):
        raw = "DE-AC01-91RW00134, NUDOCS 9801210005"
        self.assertEqual(references({"DocumentReportNumber": [raw]}), ({"9801210005": raw}, 0))
        found, unparsed = references({"DocumentReportNumber": ["DRAWING A, NUDOCS 8510110150, NUDOCS 8510110153", "NUDOCS 771190160", "NUDOCS 027219", "Report referencing NUDOCS 0000000001", "NUDOCS 0000000002-extra"]})
        self.assertEqual(set(found), {"8510110150", "8510110153"})
        self.assertEqual(unparsed, 4)

    def old_map(self):
        # Canonical v1 and v2 links match for the initial fixture.
        build(self.catalog)
        source = sqlite3.connect(self.catalog); source.row_factory = sqlite3.Row
        row = source.execute("SELECT rowid,metadata FROM items WHERE record_id='id:unrelated'").fetchone()
        metadata = json.loads(zlib.decompress(row["metadata"]))
        metadata["DocumentReportNumber"] = ["DRAWING A, NUDOCS 7902060263"]
        source.execute("UPDATE items SET metadata=? WHERE rowid=?", (zlib.compress(json.dumps(metadata).encode()), row["rowid"]))
        source.commit()
        signature = snapshot(source, self.catalog, 1)
        source.close()
        target = sqlite3.connect(default_path(self.catalog))
        target.execute("UPDATE link_state SET snapshot=?", (signature,))
        target.execute("DROP TABLE legacy_link_exceptions")
        target.commit(); target.close()
        return signature

    def test_v1_upgrade_reconciles_counts_adds_only_new_links_and_is_idempotent(self):
        self.old_map()
        result = upgrade(self.catalog)
        self.assertEqual((result["state"], result["scanned"], result["added_links"], result["links"], result["unparsed"]),
                         ("upgraded", 3, 1, 4, 0))
        self.assertEqual(lookup(self.catalog, "7902060263")["records"][0]["accession"], "ML20001A004")
        self.assertEqual(lookup(self.catalog, "7902060261")["matched_records"], 2)
        self.assertEqual(upgrade(self.catalog)["state"], "already_current")

    def test_cancelled_or_low_storage_upgrade_retains_old_snapshot_and_links(self):
        signature = self.old_map()
        with self.assertRaises(InterruptedError):
            upgrade(self.catalog, cancelled=lambda: True)
        with patch("adams_rag.legacy_links.require_space", side_effect=[None, StoragePause("low")]):
            with self.assertRaises(StoragePause):
                upgrade(self.catalog)
        target = sqlite3.connect(default_path(self.catalog))
        self.assertEqual(target.execute("SELECT snapshot,links FROM link_state").fetchone(), (signature, 3))
        self.assertEqual(target.execute("SELECT count(*) FROM legacy_links").fetchone()[0], 3)
        target.close()

    def test_upgrade_rejects_changed_catalog_and_inconsistent_counts(self):
        self.old_map()
        target = sqlite3.connect(default_path(self.catalog))
        target.execute("UPDATE link_state SET links=links+1"); target.commit(); target.close()
        with self.assertRaisesRegex(ValueError, "reconciliation failed"):
            upgrade(self.catalog)
        source = sqlite3.connect(self.catalog)
        source.execute("UPDATE libraries SET finished='changed'"); source.commit(); source.close()
        with self.assertRaisesRegex(ValueError, "exact catalog"):
            upgrade(self.catalog)

    def test_complete_map_preserves_many_to_many_links_without_merging_legacy_records(self):
        state = build(self.catalog, batch_size=1)
        self.assertEqual((state["processed"], state["links"], state["unparsed"], state["state"]), (3, 3, 1, "complete"))
        result = lookup(self.catalog, "7902060261")
        self.assertEqual(result["legacy_matched_records"], 2)
        self.assertEqual(len(result["legacy_source_keys"]), 2)
        self.assertEqual({r["accession"] for r in result["records"]}, {"ML20001A001", "ML20001A002"})
        self.assertFalse(result["identity_equivalence_verified"])
        self.assertFalse(result["full_text_included"])
        self.assertEqual(lookup(self.catalog, "7902060262")["matched_records"], 1)

    def test_resume_is_atomic_and_idempotent(self):
        first = build(self.catalog, max_rows=1)
        self.assertEqual((first["state"], first["processed"]), ("batch_limit", 1))
        self.assertEqual(lookup(self.catalog, "7902060261")["matched_records"], 1)
        second = build(self.catalog)
        self.assertEqual((second["state"], second["links"]), ("complete", 3))
        self.assertEqual(build(self.catalog)["processed_this_run"], 0)

    def test_cancelled_batch_does_not_advance_cursor_or_add_links(self):
        checks = iter([False, True])
        result = build(self.catalog, cancelled=lambda: next(checks, True))
        self.assertEqual((result["state"], result["cursor"], result["links"]), ("stopped", 0, 0))
        self.assertEqual(build(self.catalog)["links"], 3)

    def test_low_storage_does_not_write_and_mid_batch_drop_preserves_checkpoint(self):
        with patch("adams_rag.legacy_links.require_space", side_effect=StoragePause("low")):
            with self.assertRaises(StoragePause):
                build(self.catalog)
        self.assertFalse(default_path(self.catalog).exists())
        with patch("adams_rag.legacy_links.require_space", side_effect=[None, None, StoragePause("low")]):
            state = build(self.catalog)
        self.assertEqual((state["state"], state["cursor"], state["links"]), ("storage_pause", 0, 0))
        self.assertEqual(build(self.catalog)["links"], 3)

    def test_snapshot_changes_reject_stale_links_and_resume(self):
        build(self.catalog)
        db = sqlite3.connect(self.catalog)
        db.execute("UPDATE libraries SET finished='2026-10-31'"); db.commit(); db.close()
        self.assertEqual(lookup(self.catalog, "7902060261")["state"], "stale")
        self.assertEqual(lookup(self.catalog, "7902060261")["records"], [])
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            build(self.catalog)

    def test_bounded_lookup_reports_truncation_and_invalid_inputs(self):
        build(self.catalog)
        result = lookup(self.catalog, "7902060261", limit=1)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["matched_records"], 2)
        self.assertTrue(result["truncated"])
        for value in ["1", "7902060261 OR 1=1", "ML20001A001", 7902060261]:
            with self.assertRaises(ValueError):
                lookup(self.catalog, value)

    def test_missing_map_retains_legacy_identity_and_never_implies_absence(self):
        result = lookup(self.catalog, "7902060261")
        self.assertEqual(result["state"], "not_built")
        self.assertEqual(result["legacy_matched_records"], 2)

    def test_live_rag_fetches_linked_main_records_and_cites_their_actual_accessions(self):
        build(self.catalog)
        store = Store(Path(self.tmp.name) / "research.sqlite")
        calls = []
        class Client:
            def request(self, payload=None, accession=None):
                self_assert = accession in {"ML20001A001", "ML20001A002"} and payload is None
                if not self_assert:
                    raise AssertionError("Only linked main-library document retrieval is expected")
                calls.append(accession)
                return {"document": {"AccessionNumber": accession, "DocumentTitle": "Inspection " + accession,
                    "DocumentReportNumber": ["NUDOCS 7902060261"], "content": "Valve inspection report " + accession,
                    "IsLegacy": "No", "IdT": accession, "DocumentDate": "1979-02-06"}}
        try:
            result = research(store, "Summarize legacy record 7902060261", catalog=self.catalog,
                              client=Client(), generate=False)
            self.assertEqual(set(calls), {"ML20001A001", "ML20001A002"})
            self.assertEqual({r["accession"] for r in result["sources"]}, set(calls))
            self.assertTrue(all(r["legacy_report_references"] == ["7902060261"] for r in result["sources"]))
            self.assertTrue(any("multiple catalog entries" in w for w in result["warnings"]))
            self.assertFalse(result["legacy_resolution"][0]["identity_equivalence_verified"])
            local = research(store, "Summarize legacy record 7902060261", catalog=self.catalog, live=False, generate=False)
            self.assertEqual({r["accession"] for r in local["sources"]}, set(calls))
            filtered = research(store, "Summarize legacy record 7902060261", catalog=self.catalog,
                                live=False, generate=False, date_from="2020-01-01")
            self.assertEqual(filtered["sources"], [])
        finally:
            store.db.close()

    def test_missing_digitization_link_cannot_substitute_unrelated_text(self):
        store = Store(Path(self.tmp.name) / "research.sqlite")
        class Client:
            def request(self, *args, **kwargs):
                raise AssertionError("No unverified numeric GetDocument or broad substitute search")
        try:
            with store.db:
                store.put({"AccessionNumber": "ML20001A009", "content": "Unrelated valve report mentioning 7902060261"})
            result = research(store, "What does legacy 7902060261 say about valve reports?",
                              catalog=self.catalog, client=Client(), generate=False)
            self.assertEqual(result["sources"], [])
            self.assertTrue(result["insufficient_evidence"])
            self.assertEqual(result["discovery"]["attempted"], 0)
            self.assertEqual(result["legacy_resolution"][0]["state"], "not_built")
        finally:
            store.db.close()


if __name__ == "__main__":
    unittest.main()
