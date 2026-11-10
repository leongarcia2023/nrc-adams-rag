import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from adams_rag.extraction_correction import apply_correction, has_reviewed_version
from adams_rag.extraction_review import for_version, record
from adams_rag.export import document_markdown
from adams_rag.ingest import import_pdf
from adams_rag.research import hybrid_retrieve
from adams_rag.storage import StoragePause
from adams_rag.store import Store, digest
from adams_rag.text_quality import get


class ExtractionCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "research.sqlite")
        self.pdf = self.root / "source.pdf"
        self.pdf.write_bytes(b"retained PDF fixture")
        self.sha = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        self.accession = "ML993430014"
        self.pages = ["Valve introduction with an omission.", "Valve table OCR: 10% and lost columns.",
                      "Valve narrative with another corrupted exponent."]
        with self.store.db:
            self.store.put({"AccessionNumber": self.accession, "DocumentTitle": "Original title",
                "DocumentDate": "1999-12-03", "IdT": "stable-id", "LocalFileSHA256": self.sha,
                "ExtractedPageCount": 3, "PageExtraction": [{"page": n, "method": "apple_vision_ocr"} for n in (1, 2, 3)]}, "pdf", self.pages)
        self.old_hash = self.store.db.execute("SELECT text_hash FROM documents").fetchone()[0]
        record(self.store, self.accession, self.old_hash, 1, "warn", ["omitted_text"], "Known omitted text on page one.")
        record(self.store, self.accession, self.old_hash, 2, "exclude", ["numeric_ocr_error", "table_layout_loss"], "Damaged table values and columns.")
        record(self.store, self.accession, self.old_hash, 3, "exclude", ["numeric_ocr_error"], "Damaged exponent on page three.")
        self.manifest = {"schema_version": 1, "accession": self.accession, "base_text_hash": self.old_hash,
            "pdf_sha256": self.sha, "reviewer": "Fixture visual review", "reviewed_utc": "2026-11-01T10:00:00+00:00",
            "pages": [{"page": 2, "before_sha256": digest(self.pages[1]),
                "text": "Valve table, multiplier 10^-6.\nSite | LLNL93 | EPRI89\n36 | 13.6 | 0.14",
                "resolved_reasons": ["numeric_ocr_error", "table_layout_loss"],
                "review_note": "Compared the complete table, multiplier, headers and rows with the retained source."}]}

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def current(self):
        return self.store.db.execute("SELECT * FROM documents WHERE accession=?", (self.accession,)).fetchone()

    def text(self, text_hash):
        return zlib.decompress(self.store.db.execute("SELECT content FROM document_versions WHERE accession=? AND text_hash=? LIMIT 1",
            (self.accession, text_hash)).fetchone()[0]).decode()

    def extract(self, pages):
        module = SimpleNamespace(PdfReader=lambda _: SimpleNamespace(pages=[
            SimpleNamespace(extract_text=lambda text=text: text) for text in pages]))
        with patch.dict("sys.modules", {"pypdf": module}):
            return import_pdf(self.store, self.pdf, self.accession)

    def test_publishes_new_version_preserving_old_text_and_other_page_reviews(self):
        result = apply_correction(self.store, self.pdf, self.manifest)
        self.assertEqual(result["corrected_pages"], [2])
        self.assertEqual(result["carried_review_pages"], [1, 3])
        self.assertEqual(self.text(self.old_hash), "\f".join(self.pages))
        new = self.text(result["text_hash"]).split("\f")
        self.assertEqual(new, [self.pages[0], self.manifest["pages"][0]["text"], self.pages[2]])
        self.assertEqual([r["page"] for r in for_version(self.store.db, self.accession, self.old_hash)], [1, 2, 3])
        self.assertEqual([r["page"] for r in for_version(self.store.db, self.accession, result["text_hash"])], [1, 3])
        self.assertEqual({s["page"] for s in hybrid_retrieve(self.store, "Valve")}, {1, 2})
        self.assertEqual(self.current()["document_date"], "1999-12-03")
        metadata = json.loads(self.current()["metadata"])
        self.assertEqual(metadata["IdT"], "stable-id")
        self.assertEqual(metadata["PageExtraction"][1]["method"], "reviewed_transcription")
        self.assertFalse(get(self.store.db, self.accession)["extraction_verified"])

    def test_same_manifest_is_idempotent(self):
        first = apply_correction(self.store, self.pdf, self.manifest)
        second = apply_correction(self.store, self.pdf, self.manifest)
        self.assertEqual(second["state"], "already_applied")
        self.assertEqual(second["text_hash"], first["text_hash"])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM extraction_corrections").fetchone()[0], 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 2)

    def test_correction_reads_a_migrated_markdown_source(self):
        from adams_rag.markdown_archive import export_batch, setup
        from adams_rag.markdown_migrate import migrate_versions
        from adams_rag.source_content import read_text
        directory = self.root / 'markdown'; directory.mkdir()
        setup(self.store.db)
        export_batch(self.store, [self.current()], directory, {self.accession:self.pdf})
        migrate_versions(self.store, dict(versions_migrated=0, historical_versions_exported=0,
            redundant_database_payload_bytes=0))
        result = apply_correction(self.store, self.pdf, self.manifest)
        old = self.store.db.execute('SELECT content FROM document_versions WHERE text_hash=?', (self.old_hash,)).fetchone()
        self.assertEqual(read_text(old[0], self.store.db), '\f'.join(self.pages))
        self.assertEqual(self.text(result['text_hash']).split('\f')[1], self.manifest['pages'][0]['text'])

    def test_exports_keep_review_provenance_bound_to_the_requested_version(self):
        old_version = self.store.db.execute("SELECT version FROM document_versions").fetchone()[0]
        result = apply_correction(self.store, self.pdf, self.manifest)
        old = document_markdown(self.store, self.accession, old_version)
        current = document_markdown(self.store, self.accession)
        self.assertIn("reviewed_page_transcriptions:", current)
        self.assertIn(result["correction_id"], current)
        self.assertIn(self.manifest["reviewer"], current)
        self.assertIn(self.sha, current)
        self.assertNotIn("reviewed_page_transcriptions:", old)
        self.assertNotIn(result["correction_id"], old)
        self.assertIn("10% and lost columns", old)
        self.assertIn("multiplier 10^-6", current)

    def test_stale_document_or_page_and_wrong_pdf_cannot_publish(self):
        for field in ("base_text_hash", "pdf_sha256", "before_sha256"):
            manifest = copy.deepcopy(self.manifest)
            target = manifest["pages"][0] if field == "before_sha256" else manifest
            target[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_correction(self.store, self.pdf, manifest)
            self.assertEqual(self.current()["text_hash"], self.old_hash)

    def test_whole_document_and_unresolved_page_issues_remain_excluded(self):
        record(self.store, self.accession, self.old_hash, 0, "exclude", ["incorrect_text"], "Whole-document issue remains unresolved.")
        self.manifest["pages"][0]["resolved_reasons"] = ["numeric_ocr_error"]
        result = apply_correction(self.store, self.pdf, self.manifest)
        reviews = {r["page"]: r for r in for_version(self.store.db, self.accession, result["text_hash"])}
        self.assertEqual(set(reviews), {0, 1, 2, 3})
        self.assertEqual(reviews[2]["reasons"], ["table_layout_loss"])
        self.assertEqual(hybrid_retrieve(self.store, "Valve"), [])

    def test_storage_pause_and_failed_publication_preserve_document_and_reviews(self):
        with patch("adams_rag.extraction_correction.require_space", side_effect=StoragePause("reserve")):
            with self.assertRaises(StoragePause):
                apply_correction(self.store, self.pdf, self.manifest)
        put = self.store.put
        def fail_after_put(*args):
            put(*args)
            raise RuntimeError("publication failure")
        with patch.object(self.store, "put", fail_after_put):
            with self.assertRaises(RuntimeError):
                apply_correction(self.store, self.pdf, self.manifest)
        self.assertEqual(self.current()["text_hash"], self.old_hash)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 1)
        self.assertFalse(has_reviewed_version(self.store.db, self.accession, self.old_hash))

    def test_automatic_reextraction_cannot_overwrite_reviewed_text(self):
        result = apply_correction(self.store, self.pdf, self.manifest)
        with self.assertRaisesRegex(ValueError, "new review"):
            self.extract(self.pages)
        self.assertEqual(self.current()["text_hash"], result["text_hash"])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 2)

    def test_identical_reextraction_keeps_review_provenance(self):
        result = apply_correction(self.store, self.pdf, self.manifest)
        self.extract(self.text(result["text_hash"]).split("\f"))
        self.assertEqual(self.current()["text_hash"], result["text_hash"])
        self.assertEqual(json.loads(self.current()["metadata"])["PageExtraction"][1]["method"], "reviewed_transcription")

    def test_changed_pdf_cannot_inherit_a_review_even_if_extracted_text_matches(self):
        result = apply_correction(self.store, self.pdf, self.manifest)
        self.pdf.write_bytes(b"different retained PDF")
        with self.assertRaisesRegex(ValueError, "new review"):
            self.extract(self.text(result["text_hash"]).split("\f"))
        self.assertEqual(self.current()["text_hash"], result["text_hash"])

    def test_invalid_page_numbers_duplicates_and_page_breaks_are_rejected(self):
        for number in (True, 0, 4):
            manifest = copy.deepcopy(self.manifest)
            manifest["pages"][0]["page"] = number
            with self.subTest(number=number), self.assertRaises(ValueError):
                apply_correction(self.store, self.pdf, manifest)
        duplicate = copy.deepcopy(self.manifest)
        duplicate["pages"] *= 2
        with self.assertRaises(ValueError):
            apply_correction(self.store, self.pdf, duplicate)
        self.manifest["pages"][0]["text"] = "Changed page\fExtra page"
        with self.assertRaises(ValueError):
            apply_correction(self.store, self.pdf, self.manifest)

    def test_replaced_chunk_vectors_are_invalidated_and_original_version_is_retained(self):
        old_chunk = self.store.db.execute("SELECT id FROM chunks WHERE page=2").fetchone()[0]
        with self.store.db:
            self.store.db.execute("INSERT INTO dense_vectors VALUES(?,?,?,?,?)", (old_chunk, "fixture", 1, b"\x00\x00\x80\x3f", "old-hash"))
        apply_correction(self.store, self.pdf, self.manifest)
        self.assertIsNone(self.store.db.execute("SELECT 1 FROM chunks WHERE id=?", (old_chunk,)).fetchone())
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM dense_vectors").fetchone()[0], 0)
        self.assertEqual(self.text(self.old_hash), "\f".join(self.pages))


if __name__ == "__main__":
    unittest.main()
