import json
from pathlib import Path
import tempfile
import unittest
import zlib

from adams_rag.evidence import SourceBindings, SourceBindingError, saved_run_view
from adams_rag.export import document_markdown
from adams_rag.extraction_correction import apply_correction
from adams_rag.page_boundaries import PageBoundaryError, retained_pdf_pages
from adams_rag.research import research
from adams_rag.store import Store, digest
from adams_rag.text_quality import migrate


class PageBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "source.sqlite")
        self.acc = "ML20003D156"
        self.pdf = self.root / "fixture.pdf"
        self.pdf.write_bytes(b"retained PDF fixture")
        self.pdf_hash = digest(self.pdf.read_text())

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def put(self, pages, **fields):
        with self.store.db:
            return self.store.put({"AccessionNumber": self.acc, "DocumentTitle": "Valve report",
                "LocalFileSHA256": self.pdf_hash, "ExtractedPageCount": len(pages), **fields}, "pdf", pages)

    def rows(self):
        return self.store.db.execute("SELECT * FROM chunks ORDER BY ordinal").fetchall()

    def remove_map_from_legacy_version(self):
        row = self.store.db.execute("SELECT * FROM document_versions").fetchone()
        metadata = json.loads(row["metadata"])
        metadata.pop("RetainedPageLengths")
        serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
        with self.store.db:
            self.store.db.execute("UPDATE document_versions SET metadata=?,version=?",
                                  (serialized, digest(row["text_hash"] + serialized)))
            self.store.db.execute("UPDATE documents SET metadata=?", (serialized,))

    def test_exact_map_keeps_embedded_form_feeds_unicode_and_empty_pages(self):
        pages = ["alpha\fbeta Ω", "", "valve 12\f"]
        body = "\f".join(pages)
        metadata = {"RetainedPageLengths": [len(p) for p in pages], "ExtractedPageCount": 3}
        self.assertEqual(retained_pdf_pages(body, metadata), pages)

    def test_invalid_maps_and_ambiguous_old_counts_fail_closed(self):
        for metadata in ({"RetainedPageLengths": [True]}, {"RetainedPageLengths": [-1]},
                         {"RetainedPageLengths": [2]}, {"RetainedPageLengths": [1, 2]},
                         {"RetainedPageLengths": [3], "ExtractedPageCount": 2},
                         {"ExtractedPageCount": True}, {"ExtractedPageCount": 3}):
            with self.subTest(metadata=metadata), self.assertRaises(PageBoundaryError):
                retained_pdf_pages("a\fb", metadata)

    def test_legacy_one_page_keeps_control_character_as_text(self):
        self.assertEqual(retained_pdf_pages("alpha\fbeta", {"ExtractedPageCount": 1}), ["alpha\fbeta"])
        self.assertEqual(retained_pdf_pages("alpha\fbeta", {"ExtractedPageCount": 2}), ["alpha", "beta"])
        self.assertEqual(retained_pdf_pages("alpha\fbeta", {}), ["alpha", "beta"])

    def test_bad_delimiter_is_rejected_even_when_lengths_sum_matches(self):
        with self.assertRaises(PageBoundaryError):
            retained_pdf_pages("aXb", {"RetainedPageLengths": [1, 1]})

    def test_live_saved_and_exported_passages_share_physical_pages(self):
        self.put(["Valve alpha\fbeta", "Valve gamma"])
        result = research(self.store, "valve", live=False, generate=False)
        self.assertEqual({s["page"] for s in result["sources"]}, {1, 2})
        self.assertIn("Valve alpha\fbeta", [s["text"] for s in result["sources"]])
        self.assertTrue(all(s["source_version"]["state"] == "matched_retained_text" for s in result["sources"]))
        view = saved_run_view(self.store, result["id"])
        self.assertEqual({s["state"] for s in view["review_updates"]["sources"]}, {"no_matching_review"})
        exported = document_markdown(self.store, self.acc)
        self.assertEqual(exported.count("## PDF page "), 2)
        self.assertIn("Valve alpha\fbeta", exported)

    def test_old_one_page_version_binds_without_rewriting_source(self):
        self.put(["Valve alpha\fbeta"])
        self.remove_map_from_legacy_version()
        before = tuple(self.store.db.execute("SELECT version,text_hash,content FROM document_versions").fetchone())
        row = dict(self.rows()[0])
        _, binding = SourceBindings(self.store.db).bind(row)
        self.assertEqual(binding["page"], 1)
        self.assertEqual(binding["end_character"], len(row["text"]))
        self.assertEqual(tuple(self.store.db.execute("SELECT version,text_hash,content FROM document_versions").fetchone()), before)

    def test_ambiguous_old_multipage_version_cannot_be_bound_or_exported(self):
        self.put(["Valve alpha\fbeta", "Valve gamma"])
        self.remove_map_from_legacy_version()
        with self.assertRaisesRegex(SourceBindingError, "ambiguous"):
            SourceBindings(self.store.db).bind(dict(self.rows()[0]))
        with self.assertRaisesRegex(PageBoundaryError, "ambiguous"):
            document_markdown(self.store, self.acc)

    def test_same_body_with_changed_page_layout_rebuilds_chunks_and_preserves_old_run(self):
        self.put(["Valve alpha\fbeta", "gamma"])
        result = research(self.store, "valve", live=False, generate=False)
        before = [r["id"] for r in self.rows()]
        old_hash = self.store.db.execute("SELECT text_hash FROM documents").fetchone()[0]
        self.put(["Valve alpha", "beta\fgamma"])
        self.assertEqual(self.store.db.execute("SELECT text_hash FROM documents").fetchone()[0], old_hash)
        self.assertTrue(set(before).isdisjoint({r["id"] for r in self.rows()}))
        self.assertEqual([(r["page"], r["text"]) for r in self.rows()], [(1, "Valve alpha"), (2, "beta\fgamma")])
        view = saved_run_view(self.store, result["id"])
        self.assertEqual(view["sources"], result["sources"])
        self.assertEqual({s["state"] for s in view["review_updates"]["sources"]}, {"no_matching_review"})

    def test_metadata_refresh_cannot_replace_retained_page_map(self):
        self.put(["Valve alpha\fbeta", "Valve gamma"])
        with self.store.db:
            self.store.put({"AccessionNumber": self.acc, "RetainedPageLengths": [999]})
        metadata = json.loads(self.store.db.execute("SELECT metadata FROM documents").fetchone()[0])
        self.assertEqual(metadata["RetainedPageLengths"], [len("Valve alpha\fbeta"), len("Valve gamma")])
        self.assertEqual(len(research(self.store, "valve", live=False, generate=False)["sources"]), 2)

    def test_contradictory_page_count_does_not_publish(self):
        self.put(["Original valve"])
        before = tuple(self.store.db.execute("SELECT text_hash,metadata FROM documents").fetchone())
        with self.assertRaises(PageBoundaryError):
            self.put(["Replacement valve"], ExtractedPageCount=2)
        self.assertEqual(tuple(self.store.db.execute("SELECT text_hash,metadata FROM documents").fetchone()), before)

    def test_quality_migration_keeps_one_physical_page(self):
        self.put(["Valve alpha\fbeta"])
        self.remove_map_from_legacy_version()
        with self.store.db:
            self.store.db.execute("DELETE FROM document_quality")
        result = migrate(self.store)
        self.assertEqual(result["assessed"], 1)
        report = json.loads(self.store.db.execute("SELECT report FROM document_quality").fetchone()[0])
        self.assertEqual(report["extracted_pages"], 1)

    def test_correcting_second_page_does_not_split_control_character_in_first(self):
        pages = ["Valve alpha\fbeta", "Valve old value"]
        self.put(pages)
        old_hash = self.store.db.execute("SELECT text_hash FROM documents").fetchone()[0]
        manifest = {"schema_version": 1, "accession": self.acc, "base_text_hash": old_hash,
            "pdf_sha256": self.pdf_hash, "reviewer": "Fixture reviewer", "reviewed_utc": "2026-11-01T12:00:00+00:00",
            "pages": [{"page": 2, "before_sha256": digest(pages[1]), "text": "Valve reviewed value 12",
                       "resolved_reasons": ["incorrect_text"], "review_note": "Fixture transcription checked."}]}
        apply_correction(self.store, self.pdf, manifest)
        self.assertEqual([(r["page"], r["text"]) for r in self.rows()], [(1, pages[0]), (2, "Valve reviewed value 12")])
        current = self.store.db.execute("SELECT metadata FROM documents").fetchone()[0]
        self.assertEqual(json.loads(current)["RetainedPageLengths"], [len(pages[0]), len("Valve reviewed value 12")])


if __name__ == "__main__":
    unittest.main()
