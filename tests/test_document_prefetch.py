import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import zlib

from adams_rag.archive_worker import main
from adams_rag.backfill import backfill
from adams_rag.client import AdamsHTTPError, BudgetStop, RateGate
from adams_rag.document_prefetch import DocumentPrefetchClient
from adams_rag.store import Store


class DocumentPrefetchTests(unittest.TestCase):
    accessions = ["ML24017A120", "ML24017A121", "ML24017A122"]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "research.sqlite")
        self.rate = Store(self.root / "rate.sqlite")
        self.catalog = self.root / "catalog.sqlite"
        source = sqlite3.connect(self.catalog)
        source.executescript("CREATE TABLE libraries(library,final_count,finished,state);"
            "INSERT INTO libraries VALUES('main',3,'2026-10-29','complete');"
            "CREATE TABLE items(library,accession,is_package,metadata);")
        for accession in self.accessions:
            source.execute("INSERT INTO items VALUES('main',?,0,?)", (accession,
                zlib.compress(json.dumps({"AccessionNumber": accession, "DocketNumber": ["05000373"]}).encode())))
        source.commit()
        source.close()
        self.calls, self.thread_ids, self.clients = [], set(), []
        self.second_started, self.release = threading.Event(), threading.Event()
        self.third_started = threading.Event()
        self.transport_events = []
        self.release.set()
        self.error_status = self.wrong_accession = None
        owner = self

        class Transport:
            def __init__(self):
                self.thread = threading.get_ident()
                owner.transport_events.append(("created", self.thread))

            def __call__(self, request, **kwargs):
                owner.assertEqual(self.thread, threading.get_ident())
                accession = request.full_url.rsplit("/", 1)[-1]
                owner.calls.append(accession)
                owner.thread_ids.add(threading.get_ident())
                if accession == owner.accessions[2]:
                    owner.third_started.set()
                if accession == owner.accessions[1]:
                    owner.second_started.set()
                    if not owner.release.wait(3):
                        raise AssertionError("Writer did not release the lookahead fixture")
                    if owner.error_status:
                        raise urllib.error.HTTPError(request.full_url, owner.error_status,
                            "fixture", {"Retry-After": "120"}, None)
                    accession = owner.wrong_accession or accession
                return io.BytesIO(json.dumps({"document": {
                    "AccessionNumber": accession, "content": "Retained evidence about valves."}}).encode())

            def close(self):
                owner.assertEqual(self.thread, threading.get_ident())
                owner.transport_events.append(("closed", self.thread))

        self.patcher = patch("adams_rag.document_prefetch.PersistentOpener", Transport)
        self.patcher.start()

    def client(self, budget=100, cancelled=None, depth=1, rps=1000, daily_limit=1000):
        client = DocumentPrefetchClient(self.rate, "fake-key", budget, rps, daily_limit,
            storage_path=self.root, min_free_gb=.5, cancelled=cancelled, lookahead_depth=depth)
        self.clients.append(client)
        return client

    def tearDown(self):
        self.release.set()
        for client in self.clients:
            client.close()
        self.patcher.stop()
        self.rate.db.close()
        self.store.db.close()
        self.temp.cleanup()

    def test_next_download_overlaps_current_write_with_one_network_thread(self):
        self.release.clear()
        client = self.client()
        put = self.store.put
        writes = []
        def checked_put(doc, provenance):
            if doc["AccessionNumber"] == self.accessions[0]:
                self.assertTrue(self.second_started.wait(3), "Expected next request while writer handles the first")
                self.assertNotEqual(next(iter(self.thread_ids)), threading.get_ident())
                self.release.set()
            writes.append(doc["AccessionNumber"])
            return put(doc, provenance)
        with patch.object(self.store, "put", checked_put):
            result = backfill(self.store, client, self.catalog, max_documents=3)
        self.assertEqual((result["indexed"], result["last_catalog_row"]), (3, 3))
        self.assertEqual(self.calls, self.accessions)
        self.assertEqual(writes, self.accessions)
        self.assertEqual(len(self.thread_ids), 1)
        self.assertEqual(client.stats["lookahead_hits"], 2)
        self.assertEqual(self.rate.db.execute("SELECT count(*) FROM http_requests").fetchone()[0], 3)
        self.assertEqual(client.remaining, 97)
        self.assertEqual(len(self.store.lexical("valves", docket="05000373")), 3)

    def test_wrong_prefetched_identity_preserves_last_committed_cursor(self):
        self.wrong_accession = "ML24017A999"
        with self.assertRaisesRegex(ValueError, "identifier"):
            backfill(self.store, self.client(), self.catalog, max_documents=3)
        self.assertEqual(self.store.db.execute("SELECT last_row FROM backfill_cursors").fetchone()[0], 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM chunks").fetchone()[0], 1)
        self.assertEqual(self.calls, self.accessions[:2])

    def test_one_document_batch_never_prefetches_past_its_boundary(self):
        client = self.client()
        result = backfill(self.store, client, self.catalog, max_documents=1)
        self.assertEqual((result["indexed"], result["last_catalog_row"]), (1, 1))
        self.assertEqual(self.calls, self.accessions[:1])
        self.assertEqual(client.stats["lookahead_hits"], 0)

    def test_already_indexed_following_row_is_not_fetched(self):
        with self.store.db:
            self.store.put({"AccessionNumber": self.accessions[1], "content": "Existing evidence"}, "api_document_unverified")
        result = backfill(self.store, self.client(), self.catalog, max_documents=2)
        self.assertEqual((result["indexed"], result["already_indexed"]), (2, 1))
        self.assertEqual(self.calls, [self.accessions[0], self.accessions[2]])

    def test_request_budget_includes_lookahead_and_does_not_skip_unfetched_row(self):
        client = self.client(budget=1)
        result = backfill(self.store, client, self.catalog, max_documents=3)
        self.assertEqual((result["state"], result["last_catalog_row"]), ("budget_pause", 1))
        self.assertEqual(self.calls, self.accessions[:1])
        self.assertEqual(client.remaining, 0)
        self.assertEqual(self.rate.db.execute("SELECT count(*) FROM http_requests").fetchone()[0], 1)

    def test_prefetched_throttle_persists_cooldown_without_skipping_record(self):
        self.error_status = 429
        result = backfill(self.store, self.client(), self.catalog, max_documents=3)
        self.assertEqual((result["state"], result["last_catalog_row"]), ("budget_pause", 1))
        self.assertEqual(self.calls, self.accessions[:2])
        self.assertGreater(self.rate.db.execute("SELECT next_at FROM http_state").fetchone()[0], time.time()+100)
        self.assertGreater(self.rate.db.execute("SELECT slowed_interval FROM http_pacing").fetchone()[0], .001)
        self.assertEqual(self.rate.db.execute("SELECT count(*) FROM http_requests").fetchone()[0], 2)

    def test_prefetched_missing_document_is_a_durable_gap(self):
        self.error_status = 404
        result = backfill(self.store, self.client(), self.catalog, max_documents=3)
        self.assertEqual((result["indexed"], result["unavailable"], result["last_catalog_row"]), (2, 1, 3))
        self.assertEqual(self.calls, self.accessions)
        self.assertEqual(self.store.db.execute("SELECT outcome FROM backfill_records WHERE catalog_row=2").fetchone()[0], "unavailable")

    def test_prefetched_authentication_failure_cannot_skip_the_record(self):
        self.error_status = 401
        with self.assertRaises(AdamsHTTPError):
            backfill(self.store, self.client(), self.catalog, max_documents=3)
        self.assertEqual(self.store.db.execute("SELECT last_row FROM backfill_cursors").fetchone()[0], 1)
        self.assertEqual(self.calls, self.accessions[:2])

    def test_storage_drop_before_network_reservation_returns_resumable_pause(self):
        client = self.client()
        free = SimpleNamespace(free=3_000_000_000)
        def fill(*_):
            free.free = 510_000_000
        with patch("shutil.disk_usage", return_value=free):
            result = backfill(self.store, client, self.catalog, max_documents=3,
                              min_free_gb=.5, progress=fill)
        self.assertEqual((result["state"], result["last_catalog_row"]), ("storage_pause", 0))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.rate.db.execute("SELECT count(*) FROM http_requests").fetchone()[0], 0)

    def test_low_lookahead_headroom_falls_back_to_serial_without_extra_request(self):
        client = self.client()
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=600_000_000)):
            self.assertTrue(client.prefetch(self.accessions[0]))
            result = client.request(self.accessions[0])
        self.assertEqual(result["document"]["AccessionNumber"], self.accessions[0])
        self.assertEqual(self.calls, self.accessions[:1])
        self.assertEqual((client.stats["serial_fallbacks"], client.remaining), (1, 99))

    def test_storage_drop_with_pending_response_preserves_uncommitted_row(self):
        self.release.clear()
        client = self.client()
        free = SimpleNamespace(free=3_000_000_000)
        put = self.store.put
        def put_then_fill(doc, provenance):
            self.assertTrue(self.second_started.wait(3))
            value = put(doc, provenance)
            free.free = 510_000_000
            self.release.set()
            return value
        with patch("shutil.disk_usage", return_value=free), patch.object(self.store, "put", put_then_fill):
            result = backfill(self.store, client, self.catalog, max_documents=3, min_free_gb=.5)
            client.close()
        self.assertEqual((result["state"], result["last_catalog_row"]), ("storage_pause", 1))
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM documents").fetchone()[0], 1)

    def test_cancellation_discards_lookahead_but_resume_refetches_same_row(self):
        self.release.clear()
        stop = threading.Event()
        client = self.client(cancelled=stop.is_set)
        put = self.store.put
        def put_then_stop(doc, provenance):
            self.assertTrue(self.second_started.wait(3))
            value = put(doc, provenance)
            stop.set()
            self.release.set()
            return value
        with patch.object(self.store, "put", put_then_stop):
            result = backfill(self.store, client, self.catalog, max_documents=3, cancelled=stop.is_set)
        client.close()
        self.assertEqual((result["state"], result["last_catalog_row"]), ("stopped", 1))
        resumed = backfill(self.store, self.client(), self.catalog, max_documents=2)
        self.assertEqual((resumed["indexed"], resumed["last_catalog_row"]), (2, 3))
        self.assertEqual(self.calls.count(self.accessions[1]), 2)
        self.assertIsNone(client.worker_store)

    def test_unexpected_request_does_not_reuse_other_accession_response(self):
        client = self.client()
        client.prefetch(self.accessions[1])
        self.assertTrue(self.second_started.wait(3))
        result = client.request(self.accessions[2])
        self.assertEqual(result["document"]["AccessionNumber"], self.accessions[2])
        self.assertEqual(client.stats["discarded"], 1)

    def test_continuous_worker_can_enable_lookahead_and_finish_saved_catalog(self):
        args = ["--db", str(self.store.path), "--catalog", str(self.catalog),
                "--rate-db", str(self.rate.path), "--directory", str(self.root / "archive"),
                "--library", "main", "--batch-size", "3", "--rps", "10", "--min-free-gb", "0.5", "--prefetch"]
        with patch("adams_rag.archive_worker.load_api_key", return_value="fake-key"), patch("builtins.print"):
            main(args)
        status = json.loads((self.root / "archive/STATUS.json").read_text())
        self.assertEqual((status["state"], status["new_documents_this_run"]), ("scan_finished", 3))
        self.assertTrue(status["document_lookahead"])
        self.assertEqual(status["reserve_bytes"], 500_000_000)
        self.assertEqual(status["local_rps_ceiling"], 10)
        self.assertEqual(self.calls, self.accessions)
        self.assertFalse(status["full_archive_verified"])

    def test_two_requests_overlap_but_writes_remain_in_catalog_order(self):
        self.release.clear()
        client = self.client(depth=2)
        put, writes = self.store.put, []
        def ordered_put(doc, provenance):
            if doc["AccessionNumber"] == self.accessions[0]:
                # Request two cannot finish until request three has arrived.
                self.assertTrue(self.second_started.wait(3))
                self.assertTrue(self.third_started.wait(3))
                self.assertEqual(len(self.thread_ids), 2)
                self.release.set()
            writes.append(doc["AccessionNumber"])
            return put(doc, provenance)
        with patch.object(self.store, "put", ordered_put):
            result = backfill(self.store, client, self.catalog, max_documents=3)
        client.close()
        self.assertEqual((result["indexed"], result["last_catalog_row"]), (3, 3))
        self.assertEqual(writes, self.accessions)
        self.assertCountEqual(self.calls, self.accessions)
        self.assertEqual(client.stats["maximum_pending"], 2)
        self.assertTrue(all(lane.store is None for lane in client.lanes))
        self.assertEqual({thread for action, thread in self.transport_events if action == "created"},
                         {thread for action, thread in self.transport_events if action == "closed"})

    def test_two_slots_do_not_fetch_past_remaining_batch_allowance(self):
        client = self.client(depth=2)
        result = backfill(self.store, client, self.catalog, max_documents=2)
        client.close()
        self.assertEqual((result["indexed"], result["last_catalog_row"]), (2, 2))
        self.assertEqual(self.calls, self.accessions[:2])
        self.assertEqual(client.stats["maximum_pending"], 1)

    def test_two_slots_do_not_publish_later_response_after_identity_failure(self):
        self.wrong_accession = "ML24017A999"
        client = self.client(depth=2)
        with self.assertRaisesRegex(ValueError, "identifier"):
            backfill(self.store, client, self.catalog, max_documents=3)
        client.close()
        self.assertEqual(self.store.db.execute("SELECT last_row FROM backfill_cursors").fetchone()[0], 1)
        self.assertEqual([r[0] for r in self.store.db.execute("SELECT accession FROM documents")], self.accessions[:1])

    def test_two_slots_share_one_request_budget_and_resume_the_missing_prefix(self):
        client = self.client(budget=2, depth=2)
        result = backfill(self.store, client, self.catalog, max_documents=3)
        client.close()
        self.assertEqual(result["state"], "budget_pause")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(client.remaining, 0)
        self.assertEqual(self.rate.db.execute("SELECT count(*) FROM http_requests").fetchone()[0], 2)
        # Either lane can reserve the second request, but only a contiguous
        # validated prefix may be committed before the shared budget runs out.
        self.assertIn(result["last_catalog_row"], (1, 2))
        stored = [r[0] for r in self.store.db.execute("SELECT accession FROM documents ORDER BY accession")]
        self.assertEqual(stored, self.accessions[:result["last_catalog_row"]])
        resumed = backfill(self.store, self.client(depth=2), self.catalog, max_documents=3)
        self.assertEqual(resumed["last_catalog_row"], 3)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM documents WHERE status='indexed'").fetchone()[0], 3)

    def test_two_slots_share_daily_budget_and_pacing(self):
        client = self.client(depth=2, rps=20, daily_limit=2)
        result = backfill(self.store, client, self.catalog, max_documents=3)
        client.close()
        self.assertEqual(result["state"], "budget_pause")
        times = [r[0] for r in self.rate.db.execute("SELECT at FROM http_requests ORDER BY at")]
        self.assertEqual(len(times), 2)
        self.assertGreaterEqual(times[1]-times[0], .049)
        self.assertEqual(len(self.calls), 2)

    def test_throttle_in_one_lane_blocks_new_requests_in_both(self):
        self.error_status = 429
        client = self.client(depth=2)
        result = backfill(self.store, client, self.catalog, max_documents=3)
        client.close()
        self.assertEqual((result["state"], result["last_catalog_row"]), ("budget_pause", 1))
        before = len(self.calls)
        with self.assertRaises(BudgetStop):
            self.client(depth=2).request(self.accessions[2])
        self.assertEqual(len(self.calls), before)
        self.assertGreater(self.rate.db.execute("SELECT next_at FROM http_state").fetchone()[0], time.time()+100)

    def test_two_slots_fall_back_to_serial_when_extra_headroom_is_unavailable(self):
        client = self.client(depth=2)
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=800_000_000)):
            self.assertTrue(client.prefetch(self.accessions[0]))
            self.assertFalse(client.prefetch(self.accessions[0]))
            self.assertTrue(client.prefetch(self.accessions[1]))
            self.assertFalse(client.prefetch(self.accessions[2]))
            for accession in self.accessions[:2]:
                self.assertEqual(client.request(accession)["document"]["AccessionNumber"], accession)
        self.assertEqual(self.calls, self.accessions[:2])
        self.assertEqual(client.stats["serial_fallbacks"], 2)
        self.assertEqual(client.remaining, 98)

    def test_two_pending_responses_are_discarded_on_cancel_and_refetched_on_resume(self):
        self.release.clear()
        stop = threading.Event()
        client = self.client(depth=2, cancelled=stop.is_set)
        put = self.store.put
        def put_then_stop(doc, provenance):
            self.assertTrue(self.second_started.wait(3))
            self.assertTrue(self.third_started.wait(3))
            value = put(doc, provenance)
            stop.set()
            self.release.set()
            return value
        with patch.object(self.store, "put", put_then_stop):
            result = backfill(self.store, client, self.catalog, max_documents=3, cancelled=stop.is_set)
        client.close()
        self.assertEqual((result["state"], result["last_catalog_row"]), ("stopped", 1))
        resumed = backfill(self.store, self.client(depth=2), self.catalog, max_documents=2)
        self.assertEqual((resumed["indexed"], resumed["last_catalog_row"]), (2, 3))
        self.assertEqual(self.calls.count(self.accessions[1]), 2)
        self.assertEqual(self.calls.count(self.accessions[2]), 2)

    def test_storage_drop_with_two_responses_keeps_only_the_committed_prefix(self):
        self.release.clear()
        client = self.client(depth=2)
        free = SimpleNamespace(free=3_000_000_000)
        put = self.store.put
        def put_then_fill(doc, provenance):
            self.assertTrue(self.second_started.wait(3))
            self.assertTrue(self.third_started.wait(3))
            value = put(doc, provenance)
            free.free = 510_000_000
            self.release.set()
            return value
        with patch('shutil.disk_usage', return_value=free), patch.object(self.store, 'put', put_then_fill):
            result = backfill(self.store, client, self.catalog, max_documents=3, min_free_gb=.5)
            client.close()
        self.assertEqual((result['state'], result['last_catalog_row']), ('storage_pause', 1))
        self.assertEqual([r[0] for r in self.store.db.execute('SELECT accession FROM documents')], self.accessions[:1])
        self.assertTrue(all(lane.tail is None for lane in client.lanes))

    def test_extra_headroom_is_rechecked_inside_the_shared_rate_reservation(self):
        free = SimpleNamespace(free=3_000_000_000)
        original = RateGate.wait
        def fill_before_reservation(gate, *args, **kwargs):
            free.free = 800_000_000
            return original(gate, *args, **kwargs)
        client = self.client(depth=2)
        with patch('shutil.disk_usage', return_value=free), patch.object(RateGate, 'wait', fill_before_reservation):
            client.prefetch(self.accessions[0])
            self.assertEqual(client.request(self.accessions[0])['document']['AccessionNumber'], self.accessions[0])
        self.assertEqual(client.stats['serial_fallbacks'], 1)
        self.assertEqual(self.calls, self.accessions[:1])
        self.assertEqual(self.rate.db.execute('SELECT count(*) FROM http_requests').fetchone()[0], 1)

    def test_invalid_depth_rejected_before_network_resources_are_created(self):
        for depth in (0, 3, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                self.client(depth=depth)
        self.assertEqual(self.transport_events, [])

    def test_continuous_worker_reports_two_slot_configuration(self):
        args = ["--db", str(self.store.path), "--catalog", str(self.catalog),
                "--rate-db", str(self.rate.path), "--directory", str(self.root / "archive"),
                "--library", "main", "--batch-size", "3", "--rps", "6", "--min-free-gb", "0.5",
                "--prefetch", "--prefetch-depth", "2"]
        with patch("adams_rag.archive_worker.load_api_key", return_value="fake-key"), patch("builtins.print"):
            main(args)
        status = json.loads((self.root / "archive/STATUS.json").read_text())
        self.assertEqual((status["state"], status["new_documents_this_run"]), ("scan_finished", 3))
        self.assertEqual(status["document_lookahead_depth"], 2)
        self.assertEqual(status["reserve_bytes"], 500_000_000)


if __name__ == "__main__":
    unittest.main()
