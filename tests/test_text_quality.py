import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adams_rag.export import document_markdown
from adams_rag.research import research
from adams_rag.storage import StoragePause
from adams_rag.store import Store
from adams_rag.text_quality import UNAVAILABLE_NOTICE, assess, get, is_unavailable_notice, migrate, save


class TextQualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "research.sqlite")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def put(self, accession="ML24017A120", text="Valve testing was requested.", **kwargs):
        provenance = kwargs.pop("provenance", "api_document_unverified")
        pages = kwargs.pop("pages", None)
        with self.store.db:
            return self.store.put({"AccessionNumber": accession, "content": text, **kwargs}, provenance, pages)

    def test_text_presence_never_proves_completeness_or_accuracy(self):
        report = assess("This is readable source text.", {}, "api_document_unverified", "indexed")
        self.assertEqual(report["state"], "text_present_unverified")
        self.assertFalse(report["source_complete_verified"])
        self.assertFalse(report["extraction_verified"])
        self.assertFalse(report["page_mapping_available"])

    def test_blank_pdf_pages_remain_visible_without_discarding_useful_text(self):
        self.put(provenance="pdf", pages=["Valve evidence on page one.", "", "Valve evidence on page three."], EstimatedPageCount="3")
        report = get(self.store.db, "ML24017A120")
        self.assertEqual(report["state"], "needs_review")
        self.assertEqual(report["blank_pages"], [2])
        self.assertEqual(report["extracted_pages"], 3)
        result = research(self.store, "valve", live=False, generate=False)
        self.assertTrue(result["sources"])
        self.assertEqual({s["page"] for s in result["sources"]}, {1, 3})
        self.assertTrue(any("PDF pages yielded no text" in w for w in result["warnings"]))

    def test_sparse_body_and_damaged_characters_are_review_flags(self):
        self.put(text="Valve \ufffd \ufffd \ufffd \ufffd \ufffd", EstimatedPageCount="50")
        report = get(self.store.db, "ML24017A120")
        self.assertEqual(set(report["flags"]), {"sparse_text_for_page_count", "replacement_characters"})
        self.assertEqual(report["state"], "needs_review")

    def test_sections_are_not_compared_to_whole_document_page_counts(self):
        self.put(section_id="4", EstimatedPageCount="500")
        report = get(self.store.db, "ML24017A120")
        self.assertEqual(report["scope"], "section")
        self.assertNotIn("sparse_text_for_page_count", report["flags"])

    def test_empty_and_discovery_only_records_cannot_be_certified_as_text(self):
        self.put(text="")
        self.assertEqual(get(self.store.db, "ML24017A120")["state"], "no_text")
        self.put("ML24017A121", provenance="api_search_excerpt")
        self.assertEqual(get(self.store.db, "ML24017A121")["state"], "not_evidence")
        self.assertEqual(self.store.status()["chunks"], 0)

    def test_metadata_refresh_preserves_pdf_page_diagnostics_and_rechecks_estimates(self):
        self.put(provenance="pdf", pages=["Valve report", ""], EstimatedPageCount="2")
        self.put(text="Tiny search snippet", provenance="api_search_excerpt", EstimatedPageCount="4")
        report = get(self.store.db, "ML24017A120")
        self.assertEqual(report["blank_pages"], [2])
        self.assertEqual(report["extracted_pages"], 2)
        self.assertIn("page_count_mismatch", report["flags"])
        self.assertEqual(report["words"], 2)

    def test_replacing_text_invalidates_old_diagnostics_and_exports_old_version_faithfully(self):
        self.put(text="Sparse valve text", EstimatedPageCount="50")
        version = self.store.db.execute("SELECT version FROM document_versions").fetchone()[0]
        self.put(text="Valve test findings. " * 700, EstimatedPageCount="50")
        self.assertEqual(get(self.store.db, "ML24017A120")["state"], "text_present_unverified")
        self.assertIn('"sparse_text_for_page_count"', document_markdown(self.store, "ML24017A120", version))
        self.assertNotIn('"sparse_text_for_page_count"', document_markdown(self.store, "ML24017A120"))

    def test_migration_is_resumable_and_leaves_chunks_vectors_versions_unchanged(self):
        self.put(); self.put("ML24017A121", text="")
        before = self.store.status()
        with self.store.db:
            self.store.db.execute("DELETE FROM document_quality")
        self.assertEqual(migrate(self.store, limit=1)["assessed"], 1)
        self.assertEqual(migrate(self.store, limit=10)["state"], "caught_up")
        self.assertEqual(migrate(self.store, limit=10)["assessed"], 0)
        after = self.store.status()
        for key in ["documents", "chunks", "dense_vectors_by_model", "document_versions"]:
            self.assertEqual(before[key], after[key])

    def test_old_writer_updates_invalidate_quality_without_relabeling_stale_evidence(self):
        self.put()
        with self.store.db:
            self.store.db.execute("UPDATE documents SET metadata=? WHERE accession='ML24017A120'", (json.dumps({"EstimatedPageCount": "100"}),))
        self.assertEqual(get(self.store.db, "ML24017A120")["state"], "not_assessed")
        migrate(self.store)
        self.assertIn("sparse_text_for_page_count", get(self.store.db, "ML24017A120")["flags"])

    def test_concurrent_change_cannot_publish_an_assessment_for_the_old_source(self):
        self.put()
        row = self.store.db.execute("SELECT * FROM documents").fetchone()
        report = get(self.store.db, row["accession"])
        self.put(text="A different source text")
        with self.store.db:
            changed = save(self.store.db, row["accession"], row["text_hash"], row["metadata"], report, row["provenance"])
        self.assertEqual(changed, 0)
        self.assertEqual(get(self.store.db, row["accession"])["words"], 4)

    def test_low_storage_and_cancellation_leave_assessments_pending(self):
        self.put()
        with self.store.db:
            self.store.db.execute("DELETE FROM document_quality")
        with patch("adams_rag.storage.require_space", side_effect=StoragePause("low")):
            self.assertEqual(migrate(self.store)["state"], "storage_pause")
        self.assertEqual(migrate(self.store, cancelled=lambda: True)["state"], "stopped")
        self.assertEqual(get(self.store.db, "ML24017A120")["state"], "not_assessed")

    def test_invalid_page_metadata_does_not_create_invented_page_count(self):
        for value in ["unknown", "1.5", "-2", True, 0, None]:
            report = assess("Source text", {"EstimatedPageCount": value}, "api_document_unverified", "indexed")
            self.assertIsNone(report["reported_pages"])

    def test_service_notice_never_becomes_source_passages_or_a_text_version(self):
        notice = UNAVAILABLE_NOTICE.replace(". To view", ".\u00a0\u00a0To view") + ".\n"
        self.assertTrue(is_unavailable_notice(notice))
        self.assertEqual(self.put(text=notice), "missing_text")
        self.assertEqual(self.store.status()["chunks"], 0)
        self.assertEqual(self.store.status()["document_versions"], 0)
        quality = get(self.store.db, "ML24017A120")
        self.assertEqual((quality["state"], quality["flags"]), ("no_text", ["source_unavailable_notice"]))
        metadata = json.loads(self.store.db.execute("SELECT metadata FROM documents").fetchone()[0])
        self.assertEqual(metadata["SourceContentAvailabilityNotice"], notice)

    def test_report_quoting_notice_is_preserved_as_actual_source_text(self):
        text = "The following service notice was investigated: " + UNAVAILABLE_NOTICE + "\nThe investigation found a parser issue."
        self.assertFalse(is_unavailable_notice(text))
        self.assertEqual(self.put(text=text), "indexed")
        self.assertEqual(get(self.store.db, "ML24017A120")["state"], "text_present_unverified")

    def test_notice_refresh_cannot_erase_previously_acquired_text(self):
        self.put(provenance="pdf", pages=["Verified source location with valve text."])
        before = [dict(r) for r in self.store.db.execute("SELECT * FROM chunks")]
        self.assertEqual(self.put(text=UNAVAILABLE_NOTICE), "indexed")
        self.assertEqual([dict(r) for r in self.store.db.execute("SELECT * FROM chunks")], before)
        quality = get(self.store.db, "ML24017A120")
        self.assertNotIn("source_unavailable_notice", quality["flags"])
        self.assertTrue(quality["page_mapping_available"])

    def test_encoded_notice_is_classified_after_decoding(self):
        import base64
        self.put(text=base64.b64encode(UNAVAILABLE_NOTICE.encode()).decode(), contentEncoding="base64")
        self.assertEqual(self.store.status()["chunks"], 0)
        self.assertEqual(get(self.store.db, "ML24017A120")["flags"], ["source_unavailable_notice"])

    def test_hydration_notice_becomes_a_recoverable_text_gap(self):
        from adams_rag.ingest import hydrate
        self.put(text="")
        class Client:
            def request(self, accession):
                return {"document": {"AccessionNumber": accession, "content": UNAVAILABLE_NOTICE}}
        hydrate(self.store, Client())
        self.assertEqual(self.store.status()["documents"], {"unavailable_text": 1})
        self.assertEqual(self.store.status()["chunks"], 0)

    def test_direct_assessment_of_old_notice_cannot_certify_it_as_text(self):
        result = assess(UNAVAILABLE_NOTICE, {}, "api_document_unverified", "indexed")
        self.assertEqual(result["state"], "no_text")
        self.assertEqual(result["flags"], ["source_unavailable_notice"])


if __name__ == "__main__":
    unittest.main()
