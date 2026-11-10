"""Bounded document lookahead; only the backfill writer publishes text or cursors."""
from concurrent.futures import ThreadPoolExecutor
import threading

from .client import AdamsClient, BudgetStop, RequestBudget
from .storage import StoragePause, require_space
from .store import Store
from .transport import PersistentOpener


class _LookaheadUnavailable(Exception):
    """Continue serially when extra in-memory responses lack headroom."""


class _NetworkLane:
    """A single-thread executor owns its SQLite connection and HTTP transport."""
    def __init__(self, owner, number):
        self.owner = owner
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"adams-document-{number}")
        self.store = self.client = self.transport = self.tail = None

    def fetch(self, accession, cancel, speculative):
        owner = self.owner
        def cancelled():
            if cancel.is_set() or owner.closing.is_set() or owner.cancelled():
                return True
            # Rechecked before every attempt and inside the shared rate reservation.
            require_space(owner.storage_path, owner.min_free_gb)
            if speculative:
                try:
                    # Include all lookahead bodies and Python objects, including
                    # when a retry or a rate wait delays the next attempt.
                    require_space(owner.storage_path, owner.min_free_gb,
                                  extra_bytes=owner.lookahead_depth * 256 * 1024 * 1024)
                except StoragePause:
                    raise _LookaheadUnavailable from None
            return False

        if cancelled():
            raise BudgetStop("Document request cancelled")
        if self.client is None:
            self.store = Store(owner.rate_path)
            self.transport = PersistentOpener()
            self.client = AdamsClient(self.store, owner.key, rps=owner.rps,
                daily_limit=owner.daily_limit, budget=owner.budget, opener=self.transport,
                include_search_content=False)
        self.client.cancelled = cancelled
        return self.client.request(accession=accession)

    def _close_resources(self):
        if self.transport is not None:
            self.transport.close()
        if self.store is not None:
            self.store.db.close()
        self.store = self.client = self.transport = self.tail = None

    def close(self):
        try:
            self.pool.submit(self._close_resources).result()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)


class DocumentPrefetchClient:
    def __init__(self, store, key, max_requests, rps, daily_limit, *, storage_path,
                 min_free_gb=None, cancelled=None, lookahead_depth=1):
        if type(lookahead_depth) is not int or lookahead_depth not in (1, 2):
            raise ValueError("Use one or two document lookahead slots")
        self.rate_path, self.key = store.path, key
        self.rps, self.daily_limit = rps, daily_limit
        self.storage_path, self.min_free_gb = storage_path, min_free_gb
        self.cancelled = cancelled or (lambda: False)
        self.lookahead_depth = lookahead_depth
        self.budget = RequestBudget(max_requests)
        control = AdamsClient(store, key, rps=rps, daily_limit=daily_limit, budget=self.budget)
        self.scope = control.scope
        self.closing = threading.Event()
        self.pending = []
        self.lanes = [_NetworkLane(self, number) for number in range(lookahead_depth)]
        self.next_lane = 0
        self.stats = {"submitted": 0, "lookahead_hits": 0, "discarded": 0,
                      "serial_fallbacks": 0, "maximum_pending": 0}

    @property
    def remaining(self):
        return self.budget.remaining

    @property
    def worker_store(self):
        # Retain the existing one-lane inspection interface.
        return self.lanes[0].store

    def _submit(self, accession, speculative):
        if self.closing.is_set():
            raise RuntimeError("Document prefetch client is closed")
        self._release_finished()
        lane = next((item for item in self.lanes if item.tail is None), None)
        if lane is None:
            lane = self.lanes[self.next_lane]
            self.next_lane = (self.next_lane + 1) % len(self.lanes)
        cancel = threading.Event()
        self.stats["submitted"] += 1
        lane.tail = lane.pool.submit(lane.fetch, accession, cancel, speculative)
        return accession, lane.tail, cancel

    def _release_finished(self):
        # Consumed or discarded responses must not stay alive in scheduler
        # bookkeeping. Unconsumed results remain owned by the bounded queue.
        for lane in self.lanes:
            if lane.tail is not None and lane.tail.done():
                lane.tail = None

    def prefetch(self, accession):
        if (len(self.pending) >= self.lookahead_depth or self.closing.is_set()
                or self.cancelled() or any(item[0] == accession for item in self.pending)):
            return False
        self.pending.append(self._submit(accession, True))
        self.stats["maximum_pending"] = max(self.stats["maximum_pending"], len(self.pending))
        return True

    def _discard_first(self):
        if self.pending:
            _, future, cancel = self.pending.pop(0)
            cancel.set()
            future.cancel()
            self.stats["discarded"] += 1
            self._release_finished()

    def request(self, accession):
        # The writer may skip a hinted row that became indexed meanwhile. Keep
        # a later matching response, but never assign another accession's body.
        while self.pending and self.pending[0][0] != accession:
            self._discard_first()
        if self.pending:
            _, future, _ = self.pending.pop(0)
            self.stats["lookahead_hits"] += 1
        else:
            _, future, _ = self._submit(accession, False)
        try:
            return future.result()
        except _LookaheadUnavailable:
            # Speculation made no request. Preserve the normal storage floor
            # while allowing acquisition with the smaller serial allowance.
            self.stats["serial_fallbacks"] += 1
            return self._submit(accession, False)[1].result()
        finally:
            self._release_finished()

    def close(self):
        if self.closing.is_set():
            return
        self.closing.set()
        while self.pending:
            self._discard_first()
        for lane in self.lanes:
            lane.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
