import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

from adams_rag.client import BudgetStop, RequestBudget
from adams_rag.pdf_recovery import NoRedirects, PDFExtractionPending, PDFImportLimit, RecoveryReview, download_pdf, official_url, recover, reported_size_priority
from adams_rag.storage import StoragePause
from adams_rag.store import Store


class Response(io.BytesIO):
    def __init__(self, data=b"%PDF-test", headers=None):
        super().__init__(data)
        self.status = 200
        self.headers = headers or {"Content-Length": str(len(data))}


class PDFQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "research.sqlite")
        budget = RequestBudget(10)
        gate = Mock()
        gate.wait.side_effect = lambda reserve: reserve()
        self.client = SimpleNamespace(gate=gate, reserve=budget.take, record_response=Mock())
        self.directory = self.root / "recovery"
        self.accession = "ML993420306"
        self.url = self.add(self.accession)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def add(self, accession, pages=3, size=None):
        url = f"https://www.nrc.gov/docs/{accession[:6]}/{accession}.pdf"
        with self.store.db:
            self.store.put({"AccessionNumber": accession, "Url": url, "EstimatedPageCount": pages, "DocumentSize": size})
        return url

    def imported(self, store, path, accession, **kwargs):
        metadata = json.loads(store.db.execute("SELECT metadata FROM documents WHERE accession=?", (accession,)).fetchone()[0])
        with store.db:
            store.put(metadata, "pdf", ["Recovered valve source"])
        return {"pages": 1, "ocr_pages": 1, "blank_pages_needing_review": []}

    def run_recovery(self, **kwargs):
        return recover(self.store, self.client, self.directory, **kwargs)

    def test_success_retains_pdf_and_audit_then_skips_indexed_record(self):
        requests = []
        def opener(request, **kwargs):
            requests.append(request)
            return Response()
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=self.imported):
            result = self.run_recovery(opener=opener)
            second = self.run_recovery(opener=opener)
        self.assertEqual((result["recovered"], second["attempted"], len(requests)), (1, 0, 1))
        self.assertFalse(any("key" in key.lower() for key in dict(requests[0].header_items())))
        self.assertEqual(requests[0].full_url, self.url)
        record = self.store.db.execute("SELECT * FROM pdf_recovery").fetchone()
        self.assertEqual(record["outcome"], "indexed_unverified")
        self.assertEqual(record["sha256"], hashlib.sha256(b"%PDF-test").hexdigest())
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM pdf_recovery_events").fetchone()[0], 2)
        self.assertFalse(result["full_archive_verified"])

    def test_low_storage_after_download_keeps_pdf_and_resumes_without_network(self):
        opener = Mock(side_effect=lambda *a, **k: Response())
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=StoragePause("low")):
            first = self.run_recovery(opener=opener)
        self.assertEqual(first["state"], "storage_pause")
        self.assertTrue((self.directory / (self.accession + ".pdf")).exists())
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=self.imported):
            second = self.run_recovery(opener=opener)
        self.assertEqual(second["recovered"], 1)
        self.assertEqual(opener.call_count, 1)

    def test_budget_pause_leaves_document_eligible(self):
        self.client.gate.wait.side_effect = BudgetStop("daily ceiling")
        opener = Mock()
        self.assertEqual(self.run_recovery(opener=opener)["state"], "budget_pause")
        opener.assert_not_called()
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM pdf_recovery").fetchone()[0], 0)

    def test_throttle_persists_cooldown_and_stops_the_batch(self):
        self.add("ML993430001")
        opener = Mock(side_effect=urllib.error.HTTPError(self.url, 429, "limit", {"Retry-After": "600"}, None))
        result = self.run_recovery(opener=opener)
        self.assertEqual(result["state"], "network_wait")
        self.client.gate.defer.assert_called_once_with(600)
        self.client.gate.slow_down.assert_called_once()
        self.assertEqual(opener.call_count, 1)
        self.assertGreater(self.store.db.execute("SELECT next_retry FROM pdf_recovery").fetchone()[0], time.time())

    def test_missing_pdf_is_recorded_and_next_document_can_proceed(self):
        self.add("ML993430001")
        opener = Mock(side_effect=[urllib.error.HTTPError(self.url, 404, "missing", {}, None), Response()])
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=self.imported):
            result = self.run_recovery(opener=opener)
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(self.store.db.execute("SELECT outcome FROM pdf_recovery WHERE accession=?", (self.accession,)).fetchone()[0], "http_error")

    def test_denial_stops_batch_and_changed_page_policy_does_not_bypass_cooldown(self):
        opener = Mock(side_effect=urllib.error.HTTPError(self.url, 403, "denied", {}, None))
        self.assertEqual(self.run_recovery(opener=opener)["state"], "needs_review")
        result = self.run_recovery(opener=opener, max_pages=100, order="smallest-first")
        self.assertEqual(result["attempted"], 0)
        self.assertEqual(opener.call_count, 1)

    def test_smallest_first_is_bounded_and_preserves_large_and_unknown_gaps(self):
        self.add(self.accession, size="800000000")
        self.add("ML993430001")
        self.add("ML993430002", size="20")
        requested = []
        def opener(request, **kwargs):
            requested.append(request.full_url.rsplit("/", 1)[-1][:-4])
            return Response()
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=self.imported):
            first = self.run_recovery(opener=opener, limit=1, order="smallest-first")
            self.assertEqual(requested, ["ML993430002"])
            self.assertEqual(self.store.db.execute("SELECT count(*) FROM documents WHERE status<>'indexed'").fetchone()[0], 2)
            second = self.run_recovery(opener=opener, limit=2, order="smallest-first")
        self.assertEqual(requested, ["ML993430002", self.accession, "ML993430001"])
        self.assertEqual((first["recovered"], second["recovered"]), (1, 2))

    def test_oldest_default_remains_and_smallest_ties_are_stable(self):
        self.add(self.accession, size=200)
        self.add("ML993430001", size=10)
        self.add("ML993430002", size=10)
        requested = []
        def opener(request, **kwargs):
            requested.append(request.full_url.rsplit("/", 1)[-1][:-4])
            return Response()
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=self.imported):
            self.run_recovery(opener=opener, limit=1)
            self.run_recovery(opener=opener, limit=2, order="smallest-first")
        self.assertEqual(requested, [self.accession, "ML993430001", "ML993430002"])

    def test_reported_sizes_are_hints_and_invalid_sizes_sort_as_unknown(self):
        for value in (None, "", "-1", True, 3.5, "1 MB", "0", "９", str(2**63)):
            priority = reported_size_priority({"metadata": json.dumps({"DocumentSize": value}), "queue_row": 7})
            self.assertEqual(priority, (1, 0, 7))
        self.add(self.accession, size=1)
        with patch("adams_rag.pdf_recovery.import_pdf") as importer:
            result = self.run_recovery(opener=lambda *a, **k: Response(), order="smallest-first", max_bytes=8)
        self.assertEqual(result["outcomes"][0]["outcome"], "needs_review")
        importer.assert_not_called()
        self.assertFalse((self.directory / (self.accession + ".pdf")).exists())

    def test_order_change_alone_does_not_retry_reviewed_gap(self):
        opener = Mock(side_effect=lambda *a, **k: Response())
        self.run_recovery(opener=opener, max_bytes=8)
        result = self.run_recovery(opener=opener, max_bytes=8, order="smallest-first")
        self.assertEqual(result["attempted"], 0)
        self.assertEqual(opener.call_count, 1)

    def test_changed_policy_revisits_indexed_gap_without_duplicate_selection(self):
        def incomplete(store, path, accession, **kwargs):
            result = self.imported(store, path, accession, **kwargs)
            return {**result, 'blank_pages_needing_review': [2]}
        opener = Mock(side_effect=lambda *a, **k: Response())
        with patch('adams_rag.pdf_recovery.import_pdf', side_effect=incomplete):
            first = self.run_recovery(opener=opener)
            unchanged = self.run_recovery(opener=opener, order='smallest-first')
        self.assertEqual(first['outcomes'][-1]['outcome'], 'indexed_with_gaps')
        self.assertEqual(unchanged['attempted'], 0)
        with patch('adams_rag.pdf_recovery.import_pdf', side_effect=self.imported):
            changed = self.run_recovery(opener=opener, max_pages=100, order='smallest-first')
        self.assertEqual((changed['attempted'], changed['recovered']), (1, 1))
        self.assertEqual(opener.call_count, 1)

    def test_indexed_and_missing_gap_branches_match_original_eligibility(self):
        from adams_rag.pdf_recovery import RECOVERY_CANDIDATES_SQL
        self.run_recovery(cancelled=lambda: True)
        policy = json.dumps({'version': 4, 'max_pages': 100, 'ocr': True,
                             'max_bytes': 25 * 1024**2, 'max_new_ocr_pages': 20}, sort_keys=True)
        cases = [
            ('indexed', 'indexed_with_gaps', 'old', None),
            ('indexed', 'indexed_with_gaps', policy, None),
            ('unavailable_text', 'indexed_with_gaps', 'old', None),
            ('missing_text', 'http_error', 'old', time.time() + 10000),
            ('missing_text', 'http_error', policy, 0),
            ('missing_text', 'needs_review', 'old', None),
            ('missing_text', 'needs_review', policy, None),
            ('legacy_metadata', 'indexed_with_gaps', 'old', None),
            ('indexed', 'indexed_unverified', 'old', 0),
        ]
        for number, (status, outcome, old_policy, retry) in enumerate(cases):
            accession = f'ML993430{number:03d}'
            self.add(accession, size=number + 1)
            with self.store.db:
                self.store.db.execute('UPDATE documents SET status=? WHERE accession=?', (status, accession))
                self.store.db.execute('INSERT INTO pdf_recovery(accession,outcome,policy,next_retry) VALUES(?,?,?,?)',
                                      (accession, outcome, old_policy, retry))
        parameters = (policy, time.time(), policy)
        old = """SELECT d.rowid AS queue_row,d.accession,d.metadata,p.sha256 FROM documents d
            LEFT JOIN pdf_recovery p USING(accession)
            WHERE (d.status IN ('missing_text','unavailable_text') OR
                   (p.outcome='indexed_with_gaps' AND p.policy<>?)) AND
            (p.accession IS NULL OR p.next_retry<=? OR
             (p.outcome IN ('needs_review','no_readable_text','indexed_with_gaps') AND p.policy<>?))"""
        expected = [tuple(r) for r in self.store.db.execute(old + ' ORDER BY queue_row', parameters)]
        actual = [tuple(r) for r in self.store.db.execute(RECOVERY_CANDIDATES_SQL + ' ORDER BY queue_row', parameters)]
        self.assertEqual(actual, expected)
        self.assertEqual(len({r[0] for r in actual}), len(actual))

    def test_smallest_selection_cancels_without_requests_or_gap_changes(self):
        self.add("ML993430001", size=10)
        cancel = Mock(side_effect=[False, True])
        opener = Mock()
        result = self.run_recovery(order="smallest-first", cancelled=cancel, opener=opener)
        self.assertEqual((result["state"], result["attempted"]), ("stopped", 0))
        opener.assert_not_called()
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM pdf_recovery_events").fetchone()[0], 0)

    def test_cli_passes_shared_rate_budget_and_order(self):
        from adams_rag.pdf_recovery import main
        with patch("adams_rag.config.load_api_key", return_value="test-only"), \
             patch("adams_rag.client.AdamsClient") as client, \
             patch("adams_rag.pdf_recovery.recover", return_value={"state": "batch_limit"}) as recovery, \
             patch("builtins.print"), patch("adams_rag.pdf_recovery.signal.signal"):
            main(["--db", str(self.root / "cli.sqlite"), "--rate-db", str(self.root / "rate.sqlite"),
                  "--directory", str(self.directory), "--rps", "6", "--daily-limit", "180000",
                  "--order", "smallest-first", "--limit", "10"])
        self.assertEqual(client.call_args.kwargs["rps"], 6)
        self.assertEqual(client.call_args.kwargs["daily_limit"], 180000)
        self.assertEqual(client.call_args.kwargs["max_requests"], 10)
        self.assertEqual(recovery.call_args.kwargs["order"], "smallest-first")

    def test_reported_page_estimate_does_not_exclude_a_small_actual_pdf(self):
        self.add(self.accession, pages=67)
        opener = Mock(side_effect=lambda *a, **k: Response())
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=self.imported):
            result = self.run_recovery(opener=opener)
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(opener.call_count,1)
        metadata=json.loads(self.store.db.execute('SELECT metadata FROM documents').fetchone()[0])
        self.assertEqual(metadata['EstimatedPageCount'],67)

    def test_checksum_change_never_replaces_original_audit_or_indexes_file(self):
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=StoragePause("low")):
            self.run_recovery(opener=lambda *a, **k: Response())
        (self.directory / (self.accession + ".pdf")).write_bytes(b"changed")
        with patch("adams_rag.pdf_recovery.import_pdf") as importer:
            result = self.run_recovery()
        importer.assert_not_called()
        self.assertIn("checksum changed", result["outcomes"][0]["detail"])
        self.assertEqual(self.store.db.execute("SELECT sha256 FROM pdf_recovery").fetchone()[0], hashlib.sha256(b"%PDF-test").hexdigest())

    def test_non_pdf_and_length_mismatch_do_not_publish_files(self):
        for response in [Response(b"<html>error</html>"), Response(headers={"Content-Length": "100"}), Response(headers={"Content-Length": "999999999"})]:
            with self.assertRaises(RecoveryReview):
                download_pdf(self.url, self.root / "test.pdf", self.client, opener=lambda *a, **k: response)
            self.assertFalse((self.root / "test.pdf").exists())
            self.assertFalse(list(self.root.glob("*.partial")))

    def test_space_drop_during_download_removes_partial_file(self):
        calls = [0]
        def check(*args, **kwargs):
            calls[0] += 1
            if calls[0] > 2:
                raise StoragePause("low")
        with patch("adams_rag.pdf_recovery.require_space", side_effect=check):
            with self.assertRaises(StoragePause):
                download_pdf(self.url, self.root / "test.pdf", self.client, opener=lambda *a, **k: Response())
        self.assertFalse((self.root / "test.pdf").exists())
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_foreign_or_mismatched_urls_are_not_requested(self):
        for accession, url in [(self.accession, "https://example.com/a.pdf"), ("../../bad", self.url), (self.accession, self.url + "?key=x")]:
            with self.assertRaises(RecoveryReview):
                official_url(accession, {"Url": url})
        self.assertIsNone(NoRedirects().redirect_request(None, None, 302, "redirect", {}, "https://example.com"))

    def test_cancellation_and_worker_lock_prevent_processing(self):
        import fcntl
        self.assertEqual(self.run_recovery(cancelled=lambda: True)["state"], "stopped")
        with (self.directory / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "worker is running"):
                self.run_recovery()

    def test_actual_page_limit_is_recorded_without_stopping_other_documents(self):
        self.add("ML993430001")
        def importer(store, path, accession, **kwargs):
            if accession == self.accession:
                raise PDFImportLimit("PDF has 296 pages; this import permits 1–50")
            return self.imported(store, path, accession, **kwargs)
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=importer):
            result = self.run_recovery(opener=lambda *a, **k: Response())
        self.assertEqual(result["recovered"], 1)
        self.assertIn("296 pages", self.store.db.execute("SELECT detail FROM pdf_recovery WHERE accession=?", (self.accession,)).fetchone()[0])
        self.assertEqual(self.store.status()["pdf_recovery"], {"indexed_unverified": 1, "needs_review": 1})

    def test_enabling_ocr_reconsiders_previously_blank_extraction(self):
        empty = {"pages": 1, "ocr_pages": 0, "blank_pages_needing_review": [1]}
        opener = Mock(side_effect=lambda *a, **k: Response())
        with patch("adams_rag.pdf_recovery.import_pdf", return_value=empty):
            first = self.run_recovery(opener=opener, ocr=False)
        self.assertEqual(first["recovered"], 0)
        with patch("adams_rag.pdf_recovery.import_pdf", side_effect=self.imported):
            second = self.run_recovery(opener=opener, ocr=True)
        self.assertEqual(second["recovered"], 1)
        self.assertEqual(opener.call_count, 1)

    def test_ocr_pending_retains_pdf_and_resumes_without_another_request(self):
        opener=Mock(side_effect=lambda *a,**k:Response())
        pending=PDFExtractionPending(self.accession,600,21,20)
        with patch('adams_rag.pdf_recovery.import_pdf',side_effect=pending):
            first=self.run_recovery(opener=opener,max_pages=1000)
        self.assertEqual(first['state'],'ocr_pending')
        self.assertEqual(first['recovered'],0)
        record=self.store.db.execute('SELECT * FROM pdf_recovery').fetchone()
        self.assertEqual(record['outcome'],'ocr_pending')
        self.assertEqual(json.loads(record['detail'])['next_page'],21)
        self.assertFalse(self.store.status()['chunks'])
        with patch('adams_rag.pdf_recovery.import_pdf',side_effect=self.imported):
            second=self.run_recovery(opener=opener,max_pages=1000)
        self.assertEqual(second['recovered'],1)
        self.assertEqual(opener.call_count,1)

    def test_invalid_ocr_budget_is_rejected_before_network(self):
        for allowance in (0,501,True):
            with self.assertRaises(ValueError):
                self.run_recovery(max_new_ocr_pages=allowance)
        self.client.gate.wait.assert_not_called()

    def test_file_allowance_change_retries_oversize_source_and_reaches_importer(self):
        opener=Mock(side_effect=lambda *a,**k:Response())
        first=self.run_recovery(opener=opener,max_bytes=8)
        self.assertEqual(first['outcomes'][0]['outcome'],'needs_review')
        self.assertFalse((self.directory/(self.accession+'.pdf')).exists())
        with patch('adams_rag.pdf_recovery.import_pdf',side_effect=self.imported) as importer:
            second=self.run_recovery(opener=opener,max_bytes=16)
        self.assertEqual(second['recovered'],1)
        self.assertEqual(importer.call_args.kwargs['max_bytes'],16)
        self.assertEqual(opener.call_count,2)
        self.assertEqual(json.loads(self.store.db.execute('SELECT policy FROM pdf_recovery').fetchone()[0])['max_bytes'],16)

    def test_unknown_length_body_cannot_exceed_file_allowance(self):
        response=Response(headers={'Content-Type':'application/pdf'})
        with self.assertRaises(RecoveryReview):
            download_pdf(self.url,self.root/'bounded.pdf',self.client,
                         opener=lambda *a,**k:response,max_bytes=8)
        self.assertFalse((self.root/'bounded.pdf').exists())
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_larger_download_gets_bounded_time_and_reports_actual_received_bytes(self):
        updates=[]
        with patch('adams_rag.pdf_recovery.time.monotonic',side_effect=[0,0,130,131,132]):
            result=download_pdf(self.url,self.root/'larger.pdf',self.client,
                                opener=lambda *a,**k:Response(),max_bytes=512 * 1024 * 1024,
                                progress=updates.append)
        self.assertEqual(updates[0]['bytes_received'],0)
        self.assertEqual(updates[-1]['bytes_received'],result['bytes'])
        self.assertEqual(updates[-1]['reported_bytes'],result['bytes'])
        self.assertEqual(updates[-1]['time_limit_seconds'],900)
        self.assertFalse(updates[-1]['partial_text_published'])
        with patch('adams_rag.pdf_recovery.time.monotonic',side_effect=[0,121]):
            with self.assertRaises(TimeoutError):
                download_pdf(self.url,self.root/'small.pdf',self.client,opener=lambda *a,**k:Response())
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_invalid_file_limits_are_rejected_without_requests(self):
        for size in (0,True,512 * 1024 * 1024 + 1):
            with self.assertRaises(ValueError):
                self.run_recovery(max_bytes=size)
        self.client.gate.wait.assert_not_called()

    def test_progress_is_live_and_final_state_is_published_while_lock_is_held(self):
        import fcntl
        updates=[]
        def progress(value):
            with (self.directory/'worker.lock').open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            updates.append(json.loads(json.dumps(value)))
        def importer(store,path,accession,**kwargs):
            kwargs['progress']({'accession':accession,'phase':'extracting','completed_pages':2,
                                'pages':3,'new_ocr_pages':1,'cached_ocr_pages':1,
                                'partial_text_published':False})
            raise PDFExtractionPending(accession,3,3,1)
        with patch('adams_rag.pdf_recovery.import_pdf',side_effect=importer):
            result=self.run_recovery(opener=lambda *a,**k:Response(),progress=progress)
        self.assertEqual(updates[0]['state'],'starting')
        self.assertEqual(updates[-2]['current_document']['completed_pages'],2)
        self.assertEqual(updates[-2]['state'],'running')
        self.assertEqual(updates[-1]['state'],'ocr_pending')
        self.assertNotIn('current_document',updates[-1])
        self.assertEqual(updates[-1],result)
        self.assertEqual(self.store.status()['chunks'],0)

    def test_unexpected_import_error_preserves_stage_without_exposing_exception_body(self):
        def importer(*args,**kwargs):
            kwargs['progress']({'phase':'publishing','completed_pages':779,'partial_text_published':False})
            raise ValueError('Arbitrary source content or a credential must not be logged')
        with patch('adams_rag.pdf_recovery.import_pdf',side_effect=importer):
            result=self.run_recovery(opener=lambda *a,**k:Response())
        detail=result['outcomes'][-1]['detail']; parsed=json.loads(detail)
        self.assertEqual(parsed['error_type'],'ValueError')
        self.assertEqual(parsed['last_progress']['phase'],'publishing')
        self.assertEqual(parsed['last_progress']['completed_pages'],779)
        self.assertEqual(parsed['frames'][-1]['function'],'importer')
        self.assertNotIn('credential',detail)
        self.assertEqual(self.store.db.execute('SELECT detail FROM pdf_recovery_events ORDER BY id DESC LIMIT 1').fetchone()[0],detail)


if __name__ == "__main__":
    unittest.main()
