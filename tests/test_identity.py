import json
from pathlib import Path
import tempfile
import unittest

from adams_rag.export import document_markdown
from adams_rag.identity import source_key
from adams_rag.research import hybrid_retrieve, search_live
from adams_rag.store import Store


def legacy(identity="FIRST", text="Valve inspection found overheating.", **extra):
    return {"AccessionNumber": "9901010001", "IdT": identity, "IsLegacy": "Yes",
            "DocumentTitle": "Historical inspection", "content": text, **extra}


class RecordIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "research.sqlite")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def put(self, doc, provenance="jsonl_import"):
        with self.store.db:
            return self.store.put(doc, provenance)

    def test_legacy_search_results_keep_distinct_metadata_without_creating_evidence(self):
        for identity in ["FIRST", "SECOND"]:
            self.assertEqual(self.put(legacy(identity), "api_search_excerpt"), "legacy_metadata")
        self.assertEqual(self.store.status()["discovery_records"], {"legacy": 2})
        self.assertEqual(self.store.status()["chunks"], 0)
        self.assertEqual(self.store.status()["documents"], {})
        self.assertTrue(all("content" not in json.loads(r[0]) for r in self.store.db.execute("SELECT metadata FROM discovery_records")))

    def test_full_text_imports_with_shared_accession_remain_distinct_and_searchable(self):
        self.put(legacy())
        self.put(legacy("SECOND", "The inspection found a coolant leak."))
        rows = self.store.lexical("9901010001")
        self.assertEqual({r["accession"] for r in rows}, {"nrc:legacy:id:first", "nrc:legacy:id:second"})
        sources = hybrid_retrieve(self.store, "inspection")
        self.assertEqual({r["source_record_id"] for r in sources}, {"id:first", "id:second"})
        self.assertTrue(all(r["source_accession"] == "9901010001" and r["source_library"] == "legacy" for r in sources))
        self.assertEqual(self.store.status()["document_versions"], 2)

    def test_revision_of_one_record_does_not_replace_the_other(self):
        self.put(legacy())
        self.put(legacy("SECOND", "Inspection found a coolant leak."))
        self.put(legacy(text="Inspection found corrosion."))
        self.assertFalse(self.store.lexical("overheating"))
        self.assertEqual(self.store.lexical("coolant")[0]["accession"], "nrc:legacy:id:second")
        self.assertEqual(self.store.status()["document_versions"], 3)

    def test_corrected_accession_updates_lookup_without_changing_record_identity(self):
        self.put(legacy())
        self.put(legacy(AccessionNumber="9901010002"))
        self.assertFalse(self.store.lexical("9901010001"))
        self.assertEqual(self.store.lexical("9901010002")[0]["accession"], "nrc:legacy:id:first")
        self.assertEqual(self.store.status()["documents"], {"indexed": 1})

    def test_missing_nrc_id_cannot_collapse_legacy_records(self):
        doc = legacy(); doc.pop("IdT")
        with self.assertRaisesRegex(ValueError, "record ID"):
            self.put(doc)
        self.assertEqual(self.store.status()["documents"], {})

    def test_metadata_refresh_cannot_erase_retained_legacy_text(self):
        self.put(legacy())
        self.put(legacy(text="Only a search excerpt"), "api_search_excerpt")
        self.assertTrue(self.store.lexical("overheating"))
        self.assertFalse(self.store.lexical("excerpt"))
        self.assertEqual(self.put(legacy("SECOND", text="")), "legacy_metadata")

    def test_live_legacy_discovery_does_not_call_the_accession_get_document_endpoint(self):
        class Client:
            def request(self, payload=None, accession=None):
                assert accession is None, "Legacy text requires a separately verified acquisition path"
                return {"count": 2, "results": [{"document": legacy("FIRST")}, {"document": legacy("SECOND")}]}
        result = search_live(self.store, Client(), "historical valve inspection", ["historical valve inspection"])
        self.assertEqual((result["discovered"], result["attempted"]), (2, 0))
        self.assertEqual(len(result["legacy_records"]), 2)
        self.assertTrue(result["warnings"])

    def test_export_retains_public_accession_and_unambiguous_source_key(self):
        self.put(legacy())
        output = document_markdown(self.store, source_key(legacy()))
        self.assertIn('accession: "9901010001"', output)
        self.assertIn('source_key: "nrc:legacy:id:first"', output)
        self.assertIn('nrc_record_id: "id:first"', output)
        self.assertIn('AccessionNumber=9901010001', output)


if __name__ == "__main__":
    unittest.main()
