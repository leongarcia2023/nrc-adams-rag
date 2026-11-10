import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from adams_rag.catalog import Catalog
from adams_rag.client import BudgetStop, RateGate, RequestBudget
from adams_rag.prefetch import PrefetchClient
from adams_rag.store import Store


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "rate.sqlite")
        self.catalog = Catalog(Path(self.tmp.name) / "catalog")
        self.calls, self.lock = [], threading.Lock()
        self.active, self.max_active = 0, 0
        self.both_started, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.malformed_offset = None
        owner = self

        class Transport:
            def __call__(self, request, **kwargs):
                payload = json.loads(request.data)
                library = "main" if payload["mainLibFilter"] else "legacy"
                skip = payload["skip"]
                prefix = payload["filters"][0]["value"] if payload["filters"] else "AA"
                with owner.lock:
                    owner.calls.append((library, prefix, skip))
                    owner.active += 1
                    owner.max_active = max(owner.max_active, owner.active)
                    if owner.active == 2:
                        owner.both_started.set()
                try:
                    if library == "main" and skip in (2, 4):
                        if not owner.release.wait(3):
                            raise AssertionError("Concurrent fixture did not release")
                    if library == "main" and skip == owner.malformed_offset:
                        body = {"invalid": "schema"}
                    else:
                        total = 8 if library == "main" else 0
                        body = {"count": total, "results": [{"document": {
                            "AccessionNumber": f"{prefix}{n:03}", "IdT": f"{prefix}-{n}",
                            "DocumentTitle": "fixture", "DocumentSize": "10"}}
                            for n in range(skip, min(skip + 2, total))]}
                    return io.BytesIO(json.dumps(body).encode())
                finally:
                    with owner.lock:
                        owner.active -= 1

            def close(self):
                pass

        self.patcher = patch("adams_rag.prefetch.PersistentOpener", Transport)
        self.patcher.start()
        self.client = PrefetchClient(self.store, "fake-key", 100, 1000, 1000)

    def tearDown(self):
        self.release.set()
        self.client.close()
        self.patcher.stop()
        self.catalog.db.close()
        self.store.db.close()
        self.tmp.cleanup()

    def test_two_overlapping_fetches_keep_one_validating_catalog_writer(self):
        self.release.clear()
        self.catalog.step(self.client)
        self.assertTrue(self.both_started.wait(3), "Expected both lookahead pages in flight")
        self.assertEqual(self.catalog.summary()["catalog_records"], 2)
        self.release.set()
        while self.catalog.step(self.client):
            pass
        self.assertTrue(self.catalog.reconcile(self.client))
        self.assertEqual(self.catalog.summary()["catalog_records"], 8)
        self.assertEqual(self.max_active, 2)
        self.assertEqual([s for lib, _, s in self.calls if lib == "main"].count(2), 1)
        ledger = self.store.db.execute("SELECT count(*) FROM http_requests").fetchone()[0]
        self.assertEqual(self.client.remaining, 100 - ledger)

    def test_invalid_prefetched_page_never_advances_saved_checkpoint(self):
        self.malformed_offset = 2
        self.catalog.step(self.client)
        self.catalog.step(self.client)  # Empty legacy section.
        with self.assertRaisesRegex(ValueError, "schema"):
            self.catalog.step(self.client)
        self.assertEqual(self.catalog.summary()["catalog_records"], 2)
        self.assertEqual(self.catalog.next_partition()["offset"], 2)

    def test_different_queries_cannot_reuse_cached_pages(self):
        first = [{"field": "AccessionNumber", "operator": "starts", "value": "AA"}]
        second = [{"field": "AccessionNumber", "operator": "starts", "value": "BB"}]
        self.client.search(filters=first)
        result = self.client.search(filters=second, skip=2)
        self.assertEqual([row["document"]["AccessionNumber"] for row in result["results"]], ["BB002", "BB003"])

    def test_large_sections_are_not_speculatively_deep_paged(self):
        self.client.max_leaf = 2
        self.client.search()
        self.assertEqual(self.pending_count(), 0)
        self.assertEqual(len(self.calls), 1)

    def test_count_audits_do_not_prefetch_unused_pages(self):
        self.client.search_count()
        self.assertEqual(self.pending_count(), 0)
        self.assertEqual(len(self.calls), 1)

    def pending_count(self):
        return len(self.client.pending)


class SharedLimitsTests(unittest.TestCase):
    def test_shared_budget_cannot_be_overdrawn_by_two_threads(self):
        budget, barrier = RequestBudget(1), threading.Barrier(2)
        outcomes = []
        def take():
            barrier.wait()
            try:
                budget.take()
                outcomes.append("reserved")
            except BudgetStop:
                outcomes.append("stopped")
        workers = [threading.Thread(target=take) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertCountEqual(outcomes, ["reserved", "stopped"])
        self.assertEqual(budget.remaining, 0)

    def test_rate_throttle_and_cooldown_are_shared_across_connections(self):
        with tempfile.TemporaryDirectory() as temp:
            one, two = Store(Path(temp) / "rate.sqlite"), Store(Path(temp) / "rate.sqlite")
            try:
                now = [1000.]
                sleeps = []
                def sleep(seconds):
                    sleeps.append(seconds)
                    now[0] += seconds
                a = RateGate(one, "fake-key", rps=4, clock=lambda: now[0], sleep=sleep)
                b = RateGate(two, "fake-key", rps=4, clock=lambda: now[0], sleep=sleep)
                a.wait()
                b.wait()
                self.assertEqual(sleeps, [.25])
                a.slow_down()
                self.assertEqual(b.interval, .5)
                a.defer(10)
                b.wait()
                self.assertEqual(sleeps[-1], 10)
                restarted = RateGate(two, "fake-key", rps=4, clock=lambda: now[0], sleep=sleep)
                self.assertEqual(restarted.interval, .5)
                budget = RequestBudget(0)
                with self.assertRaises(BudgetStop):
                    restarted.wait(reserve=budget.take)
                self.assertEqual(one.db.execute("SELECT count(*) FROM http_requests").fetchone()[0], 3)
            finally:
                one.db.close()
                two.db.close()


if __name__ == "__main__":
    unittest.main()
