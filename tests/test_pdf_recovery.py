import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adams_rag.export import document_markdown
from adams_rag.ingest import PDFExtractionPending, PDFImportLimit, embedded_text_recovery, import_pdf
from adams_rag.research import research
from adams_rag.storage import StoragePause
from adams_rag.store import Store
from adams_rag.text_quality import get


class PDFRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "test.sqlite")
        self.path = self.root / "source.pdf"
        self.path.write_bytes(b"pdf fixture")
        self.accession = "ML993420036"
        with self.store.db:
            self.store.put({"AccessionNumber": self.accession, "DocumentTitle": "Original title",
                            "DocumentDate": "1999-12-03", "DateAddedTimestamp": "1999-12-09",
                            "DocketNumber": ["05000321"], "IdT": "stable-record-id", "EstimatedPageCount": "3"})

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def extract(self, texts, **kwargs):
        pages = [SimpleNamespace(extract_text=lambda text=text: text) for text in texts]
        module = SimpleNamespace(PdfReader=lambda _: SimpleNamespace(pages=pages))
        with patch.dict("sys.modules", {"pypdf": module}):
            return import_pdf(self.store, self.path, self.accession, **kwargs)

    @staticmethod
    def recognized(number, text="Valve recovery text"):
        return {"page": number, "text": text, "method": "apple_vision_ocr", "lines": [{"text": text}],
                "revision": 3, "layout_verified": False}

    def test_blank_pages_only_are_recognized_with_page_citations_and_original_metadata(self):
        with patch("adams_rag.ocr.recognize_page", return_value=self.recognized(2)) as ocr:
            result = self.extract(["Embedded first page", "", "Embedded last page"], ocr=True)
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(ocr.call_args.args[1], 2)
        self.assertEqual(result["ocr_pages"], 1)
        row = self.store.db.execute("SELECT * FROM documents").fetchone()
        metadata = json.loads(row["metadata"])
        self.assertEqual(row["document_date"], "1999-12-03")
        self.assertEqual(row["title"], "Original title")
        self.assertEqual(metadata["DocketNumber"], ["05000321"])
        self.assertEqual(metadata["IdT"], "stable-record-id")
        self.assertEqual(metadata["EstimatedPageCount"], "3")
        evidence = research(self.store, "valve recovery", live=False, generate=False)
        self.assertEqual(evidence["sources"][0]["page"], 2)
        self.assertTrue(any("machine-recognized" in warning for warning in evidence["warnings"]))
        quality = get(self.store.db, self.accession)
        self.assertEqual(quality["ocr_pages"], [2])
        self.assertFalse(quality["extraction_verified"])
        self.assertIn("machine_ocr", document_markdown(self.store, self.accession))

    def test_failure_on_later_page_never_publishes_partial_document(self):
        with patch("adams_rag.ocr.recognize_page", side_effect=[self.recognized(1), RuntimeError("failed")]):
            with self.assertRaises(RuntimeError):
                self.extract(["", ""], ocr=True)
        self.assertEqual(self.store.status()["chunks"], 0)
        self.assertEqual(self.store.status()["document_versions"], 0)

    def test_nonblank_encoded_page_uses_ocr_and_retains_recovery_identity(self):
        broken = "5HYLVLRQ\x03\x15\x19\n" * 10
        with patch("adams_rag.ocr.recognize_page", return_value=self.recognized(1, "Revision 26. Quality assurance program.")) as ocr:
            result = self.extract([broken], ocr=True)
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(result['ocr_pages'], 1)
        row = self.store.db.execute('SELECT metadata FROM documents').fetchone()
        method = json.loads(row['metadata'])['PageExtraction'][0]
        self.assertEqual(method['embedded_text_recovery'], embedded_text_recovery(broken))
        self.assertEqual(method['method'], 'apple_vision_ocr')
        self.assertIn('machine_ocr', get(self.store.db, self.accession)['flags'])
        self.assertNotIn('5HYLVLRQ', document_markdown(self.store, self.accession))

    def test_nonblank_encoded_page_without_ocr_does_not_publish_partial_text(self):
        self.extract(['Retained usable text'])
        before = self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0]
        with self.assertRaisesRegex(PDFImportLimit, 'enable OCR'):
            self.extract(['New first page', '5HYLVLRQ\x03\x15\x19\n' * 10])
        self.assertEqual(self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0], before)
        self.assertEqual(self.store.status()['document_versions'], 1)

    def test_layout_whitespace_and_sparse_controls_do_not_trigger_ocr(self):
        ordinary = 'RV-493.2-A\t500 psig\n\r\f\v\x85Section 10 CFR 50.54(a).\n' * 10
        self.assertIsNone(embedded_text_recovery(ordinary))
        self.assertIsNone(embedded_text_recovery(ordinary + '\x03'))
        with patch('adams_rag.ocr.recognize_page') as ocr:
            self.extract([ordinary], ocr=True)
        ocr.assert_not_called()

    def test_still_encoded_ocr_result_is_not_published(self):
        broken = '5HYLVLRQ\x03\x15\x19\n' * 10
        with patch('adams_rag.ocr.recognize_page', return_value=self.recognized(1, broken)):
            with self.assertRaisesRegex(ValueError, 'source review is required'):
                self.extract([broken], ocr=True)
        self.assertEqual(self.store.status()['document_versions'], 0)

    def test_automatic_reextraction_cannot_clear_an_existing_exclusion(self):
        from adams_rag.extraction_review import record, for_version
        self.extract(['Known inaccurate retained page'])
        before = self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0]
        record(self.store, self.accession, before, 1, 'exclude', ['incorrect_text'], 'Requires comparison with the original page.')
        broken = '5HYLVLRQ\x03\x15\x19\n' * 10
        with patch('adams_rag.ocr.recognize_page', return_value=self.recognized(1, 'New OCR text')):
            with self.assertRaisesRegex(ValueError, 'Existing extraction reviews require a new review'):
                self.extract([broken], ocr=True)
        self.assertEqual(self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0], before)
        self.assertEqual(for_version(self.store.db, self.accession, before)[0]['disposition'], 'exclude')
        self.assertEqual(self.store.status()['document_versions'], 1)

    def test_blank_ocr_stays_a_visible_gap(self):
        with patch("adams_rag.ocr.recognize_page", return_value=self.recognized(2, "")):
            result = self.extract(["Readable first page", ""], ocr=True)
        self.assertEqual(result["blank_pages_needing_review"], [2])
        self.assertIn("blank_pdf_pages", get(self.store.db, self.accession)["flags"])

    def test_no_ocr_by_default_and_catalog_page_mismatch_is_retained(self):
        with patch("adams_rag.ocr.recognize_page") as ocr:
            result = self.extract(["First page", ""])
        ocr.assert_not_called()
        self.assertEqual(result["ocr_pages"], 0)
        self.assertIn("page_count_mismatch", get(self.store.db, self.accession)["flags"])

    def test_limits_and_storage_cancellation_leave_source_unchanged(self):
        for kwargs in [{"max_pages": 0}, {"max_pages": 5001}, {"max_pages": True}, {"max_pages": 1},
                       {"max_new_ocr_pages":0},{"max_new_ocr_pages":501},{"max_new_ocr_pages":True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.extract(["a", "b"], **kwargs)
        with self.assertRaises(InterruptedError):
            self.extract(["a"], cancelled=lambda: True)
        with patch("adams_rag.storage.require_space", side_effect=StoragePause("low")):
            with self.assertRaises(StoragePause):
                self.extract(["a"])
        self.assertEqual(self.store.status()["document_versions"], 0)

    def test_changed_file_is_rejected_before_publication(self):
        def change(*args, **kwargs):
            self.path.write_bytes(b"different PDF")
            return self.recognized(1)
        with patch("adams_rag.ocr.recognize_page", side_effect=change):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.extract([""], ocr=True)
        self.assertEqual(self.store.status()["document_versions"], 0)

    def test_explicit_metadata_overrides_preserve_record_identity(self):
        self.extract(["Source text"], title="New title", docket=[], url="https://www.nrc.gov/docs/ML9934/ML993420036.pdf")
        row = self.store.db.execute("SELECT * FROM documents").fetchone()
        self.assertEqual(row["title"], "New title")
        self.assertEqual(json.loads(row["metadata"])["DocketNumber"], [])
        self.assertEqual(json.loads(row["metadata"])["IdT"], "stable-record-id")

    def test_ocr_preflight_refuses_before_starting_worker(self):
        from adams_rag.ocr import recognize_page
        with patch("adams_rag.ocr.sys.platform", "darwin"), patch("adams_rag.ocr.importlib.util.find_spec", return_value=object()), patch("adams_rag.ocr.require_space", side_effect=StoragePause("low")), patch("adams_rag.ocr.run_guarded") as run:
            with self.assertRaises(StoragePause):
                recognize_page(self.path, 1, self.root)
        run.assert_not_called()

    def test_ocr_lock_rejects_overlapping_workers(self):
        import fcntl
        from adams_rag.ocr import recognize_page
        with (self.root / "ocr.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch("adams_rag.ocr.sys.platform", "darwin"), patch("adams_rag.ocr.importlib.util.find_spec", return_value=object()), patch("adams_rag.ocr.require_space"), patch("adams_rag.ocr.run_guarded") as run:
                with self.assertRaisesRegex(RuntimeError, "Another local OCR"):
                    recognize_page(self.path, 1, self.root)
            run.assert_not_called()

    def test_completed_pages_resume_after_later_page_failure(self):
        with patch("adams_rag.ocr.recognize_page", side_effect=[self.recognized(1), RuntimeError("interrupted")]) as ocr:
            with self.assertRaises(RuntimeError):
                self.extract(["", ""], ocr=True)
        self.assertEqual(ocr.call_count, 2)
        self.assertEqual(self.store.status()["chunks"], 0)
        with patch("adams_rag.ocr.recognize_page", return_value=self.recognized(2)) as resumed:
            result = self.extract(["", ""], ocr=True)
        self.assertEqual(resumed.call_count, 1)
        self.assertEqual(resumed.call_args.args[1], 2)
        self.assertEqual(result["ocr_pages"], 2)

    def test_changed_source_and_corrupt_cache_require_new_recognition(self):
        from adams_rag.ocr import cached_page
        with patch("adams_rag.ocr.recognize_page", return_value=self.recognized(1)) as ocr:
            cached_page(self.path, 1, self.root)
            cached_page(self.path, 1, self.root)
            self.assertEqual(ocr.call_count, 1)
            cached = next((self.root / "ocr_pages").rglob("*.gz"))
            cached.write_bytes(b"bad gzip")
            cached_page(self.path, 1, self.root)
            self.assertEqual(ocr.call_count, 2)
            self.path.write_bytes(b"updated PDF")
            cached_page(self.path, 1, self.root)
            self.assertEqual(ocr.call_count, 3)

    def test_changed_recognition_policy_invalidates_old_page_cache(self):
        from adams_rag.ocr import cached_page
        with patch("adams_rag.ocr.recognize_page", return_value=self.recognized(1)) as ocr:
            cached_page(self.path, 1, self.root)
            with patch("adams_rag.ocr.POLICY_VERSION", 2):
                cached_page(self.path, 1, self.root)
        self.assertEqual(ocr.call_count, 2)

    def test_low_space_before_page_publication_leaves_no_false_cache(self):
        from adams_rag.ocr import cached_page
        with patch("adams_rag.ocr.recognize_page", return_value=self.recognized(1)), patch("adams_rag.ocr.require_space", side_effect=StoragePause("low")):
            with self.assertRaises(StoragePause):
                cached_page(self.path, 1, self.root)
        self.assertFalse(list(self.root.rglob("*.json.gz")))

    def test_ocr_budget_resumes_after_cached_and_embedded_pages_without_partial_publication(self):
        texts=['Embedded first page','','','Embedded fourth page','']
        def recognize(path,number,*args,**kwargs):
            return self.recognized(number,f'Recovered page {number}')
        with patch('adams_rag.ocr.recognize_page',side_effect=recognize) as ocr:
            with self.assertRaises(PDFExtractionPending) as pending:
                self.extract(texts,ocr=True,max_new_ocr_pages=2)
            self.assertEqual(pending.exception.progress,{'accession':self.accession,'pages':5,
                'next_page':5,'new_ocr_pages':2,'partial_text_published':False})
            self.assertEqual([c.args[1] for c in ocr.call_args_list],[2,3])
            self.assertEqual(self.store.status()['chunks'],0)
            self.assertEqual(self.store.status()['document_versions'],0)
            ocr.reset_mock()
            result=self.extract(texts,ocr=True,max_new_ocr_pages=1)
            self.assertEqual([c.args[1] for c in ocr.call_args_list],[5])
            self.assertEqual((result['ocr_pages'],result['new_ocr_pages']),(3,1))
            ocr.reset_mock()
            again=self.extract(texts,ocr=True,max_new_ocr_pages=1)
            ocr.assert_not_called()
            self.assertEqual(again['new_ocr_pages'],0)

    def test_budget_pause_preserves_previously_indexed_evidence(self):
        self.extract(['Previously retained text'])
        old=self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0]
        with patch('adams_rag.ocr.recognize_page',return_value=self.recognized(1)):
            with self.assertRaises(PDFExtractionPending):
                self.extract(['',''],ocr=True,max_new_ocr_pages=1)
        self.assertEqual(self.store.db.execute('SELECT text_hash FROM documents').fetchone()[0],old)
        self.assertEqual(self.store.status()['document_versions'],1)

    def test_page_progress_distinguishes_cached_work_from_unpublished_text(self):
        updates=[]
        with patch('adams_rag.ocr.recognize_page',return_value=self.recognized(1)):
            with self.assertRaises(PDFExtractionPending):
                self.extract(['',''],ocr=True,max_new_ocr_pages=1,progress=updates.append)
        self.assertEqual(updates[-1]['completed_pages'],1)
        self.assertEqual(updates[-1]['new_ocr_pages'],1)
        self.assertEqual(updates[-1]['cached_ocr_pages'],0)
        self.assertTrue(all(not r['partial_text_published'] for r in updates))
        self.assertEqual(self.store.status()['chunks'],0)
        updates.clear()
        with patch('adams_rag.ocr.recognize_page',return_value=self.recognized(2)):
            self.extract(['',''],ocr=True,max_new_ocr_pages=1,progress=updates.append)
        self.assertEqual(updates[1]['cached_ocr_pages'],1)
        self.assertEqual(updates[1]['new_ocr_pages'],0)
        self.assertEqual(updates[-1]['phase'],'publishing')
        self.assertEqual(updates[-1]['completed_pages'],2)
        self.assertEqual(updates[-1]['new_ocr_pages'],1)
        self.assertEqual(updates[-1]['cached_ocr_pages'],1)
        self.assertGreater(self.store.status()['chunks'],0)

    def test_long_embedded_pdf_can_pass_the_old_500_page_boundary(self):
        result=self.extract([f'Embedded text on page {n}' for n in range(1,502)],max_pages=1000)
        self.assertEqual((result['pages'],result['new_ocr_pages']),(501,0))
        self.assertEqual(self.store.db.execute('SELECT max(page) FROM chunks').fetchone()[0],501)

    def test_source_snapshot_avoids_rehashing_each_cached_page(self):
        from adams_rag.ocr import PDFSourceSnapshot,cached_page
        from adams_rag.pdf_policy import file_sha256
        with patch('adams_rag.ocr.file_sha256',wraps=file_sha256) as hashed,patch('adams_rag.ocr.recognize_page',return_value=self.recognized(1)) as ocr:
            snapshot=PDFSourceSnapshot(self.path)
            cached_page(self.path,1,self.root,snapshot=snapshot)
            cached_page(self.path,1,self.root,snapshot=snapshot)
        self.assertEqual(hashed.call_count,1)
        self.assertEqual(ocr.call_count,1)

    def test_larger_file_requires_explicit_allowance_and_passes_it_to_ocr(self):
        with self.path.open('wb') as file:
            file.truncate(26 * 1024 * 1024)
        with self.assertRaisesRegex(ValueError,'import allowance'):
            self.extract([''],ocr=True)
        with patch('adams_rag.ocr.recognize_page',return_value=self.recognized(1)) as ocr:
            result=self.extract([''],ocr=True,max_bytes=32 * 1024 * 1024)
        self.assertEqual(result['new_ocr_pages'],1)
        self.assertEqual(ocr.call_args.kwargs['max_bytes'],32 * 1024 * 1024)
        with patch('adams_rag.ocr.recognize_page') as reused:
            self.extract([''],ocr=True,max_bytes=64 * 1024 * 1024)
        reused.assert_not_called()

    def test_invalid_file_allowance_and_native_worker_bound_fail_before_processing(self):
        from adams_rag.ocr_worker import recognize
        for size in (0,True,512 * 1024 * 1024 + 1):
            with self.assertRaises(ValueError):
                self.extract(['source'],max_bytes=size)
            with self.assertRaises(ValueError):
                recognize({'path':str(self.path),'page':1,'max_bytes':size})
        with self.assertRaisesRegex(ValueError,'allowance exceeded'):
            recognize({'path':str(self.path),'page':1,'max_bytes':1})

    def test_ocr_supervisor_passes_file_allowance_and_reserves_parse_headroom(self):
        from adams_rag.ocr import recognize_page
        with patch('adams_rag.ocr.importlib.util.find_spec',return_value=object()),patch('adams_rag.ocr.sys.platform','darwin'),patch('adams_rag.ocr.require_space') as space,patch('adams_rag.ocr.run_guarded',return_value=self.recognized(1)) as guarded:
            recognize_page(self.path,1,self.root,max_bytes=64 * 1024 * 1024)
        self.assertEqual(guarded.call_args.args[1]['max_bytes'],64 * 1024 * 1024)
        self.assertTrue(all(c.kwargs['extra_bytes'] >= 512 * 1024 * 1024 + self.path.stat().st_size * 2 for c in space.call_args_list))

    def test_same_size_source_change_or_wrong_path_rejects_cached_snapshot(self):
        from adams_rag.ocr import PDFSourceSnapshot,cached_page
        snapshot=PDFSourceSnapshot(self.path)
        with patch('adams_rag.ocr.recognize_page',return_value=self.recognized(1)):
            cached_page(self.path,1,self.root,snapshot=snapshot)
        other=self.root/'other.pdf';other.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(ValueError,'does not match'):
            cached_page(other,1,self.root,snapshot=snapshot)
        self.path.write_bytes(b'new fixture')
        with patch('adams_rag.ocr.recognize_page') as ocr:
            with self.assertRaisesRegex(ValueError,'changed during'):
                cached_page(self.path,1,self.root,snapshot=snapshot)
            ocr.assert_not_called()

    def test_ctime_only_change_rehashes_once_before_reusing_source(self):
        from adams_rag.ocr import PDFSourceSnapshot
        from adams_rag.pdf_policy import file_sha256
        snapshot=PDFSourceSnapshot(self.path)
        self.path.chmod(self.path.stat().st_mode ^ 0o100)
        self.assertEqual(snapshot._identity()[:4],snapshot.identity[:4])
        self.assertNotEqual(snapshot._identity()[4],snapshot.identity[4])
        with patch('adams_rag.ocr.file_sha256',wraps=file_sha256) as hashed:
            snapshot.check()
            snapshot.check()
        self.assertEqual(hashed.call_count,1)
        self.assertEqual(snapshot.metadata_rechecks,1)

    def test_restored_mtime_cannot_hide_changed_source_bytes(self):
        from adams_rag.ocr import PDFSourceSnapshot,PDFSourceChanged
        snapshot=PDFSourceSnapshot(self.path)
        before=self.path.stat()
        self.path.write_bytes(b'new fixture')
        os.utime(self.path,ns=(before.st_atime_ns,before.st_mtime_ns))
        self.assertEqual(snapshot._identity()[:4],snapshot.identity[:4])
        with self.assertRaises(PDFSourceChanged) as raised:
            snapshot.check()
        self.assertEqual(raised.exception.changed_fields,['ctime_ns'])
        self.assertEqual(snapshot.metadata_rechecks,0)

    def test_source_must_remain_stable_during_metadata_recheck(self):
        from adams_rag.ocr import PDFSourceSnapshot,PDFSourceChanged
        from adams_rag.pdf_policy import file_sha256
        snapshot=PDFSourceSnapshot(self.path)
        self.path.chmod(self.path.stat().st_mode ^ 0o100)
        def unstable(path,maximum):
            digest=file_sha256(path,maximum)
            path.chmod(path.stat().st_mode ^ 0o100)
            return digest
        with patch('adams_rag.ocr.file_sha256',side_effect=unstable):
            with self.assertRaises(PDFSourceChanged):
                snapshot.check()
        self.assertEqual(snapshot.metadata_rechecks,0)

    def test_source_change_during_cache_read_is_not_treated_as_cache_corruption(self):
        from adams_rag.ocr import PDFSourceSnapshot,cached_page
        snapshot=PDFSourceSnapshot(self.path)
        with patch('adams_rag.ocr.recognize_page',return_value=self.recognized(1)):
            cached_page(self.path,1,self.root,snapshot=snapshot)
        original=json.loads
        def changed(data):
            value=original(data)
            self.path.write_bytes(b'new fixture')
            return value
        with patch('adams_rag.ocr.json.loads',side_effect=changed),patch('adams_rag.ocr.recognize_page') as ocr:
            with self.assertRaisesRegex(ValueError,'changed during'):
                cached_page(self.path,1,self.root,snapshot=snapshot)
            ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
