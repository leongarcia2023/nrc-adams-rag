"""Bounded page lookahead; the catalog remains the single validating writer."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import threading

from .client import AdamsClient, BudgetStop, RequestBudget
from .store import Store
from .transport import PersistentOpener


class PrefetchClient:
    def __init__(self, store, key, max_requests, rps, daily_limit, max_leaf=1000):
        self.rate_path, self.key = store.path, key
        self.rps, self.daily_limit, self.max_leaf = rps, daily_limit, max_leaf
        self.budget = RequestBudget(max_requests)
        control = AdamsClient(store, key, max_requests, rps, daily_limit, budget=self.budget)
        self.gate, self.scope = control.gate, control.scope
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="adams-page")
        self.local = threading.local()
        self.transports, self.transport_lock = [], threading.Lock()
        self.pending = OrderedDict()
        self.stats = {"submitted": 0, "cache_hits": 0, "discarded": 0}

    @property
    def remaining(self):
        return self.budget.remaining

    @remaining.setter
    def remaining(self, value):
        self.budget.remaining = value

    def _fetch(self, kwargs, cancel):
        if cancel.is_set():
            raise BudgetStop("Speculative page request cancelled")
        if not hasattr(self.local, "transport"):
            self.local.transport = PersistentOpener()
            with self.transport_lock:
                self.transports.append(self.local.transport)
        # Each task owns its SQLite connection; no cross-thread DB operations.
        store = Store(self.rate_path)
        try:
            client = AdamsClient(store, self.key, rps=self.rps, daily_limit=self.daily_limit,
                                 opener=self.local.transport, include_search_content=False,
                                 budget=self.budget, cancelled=cancel.is_set)
            return client.search(**kwargs)
        finally:
            store.db.close()

    def _submit(self, kwargs):
        self.stats["submitted"] += 1
        cancel = threading.Event()
        return self.pool.submit(self._fetch, kwargs, cancel), cancel

    def _drop(self, key):
        self.stats["discarded"] += 1
        future, cancel = self.pending.pop(key)
        cancel.set()
        future.cancel()

    def clear(self):
        for key in list(self.pending):
            self._drop(key)

    def close(self):
        self.clear()
        self.pool.shutdown(wait=True, cancel_futures=True)
        for transport in self.transports:
            transport.close()

    def search_count(self, **kwargs):
        return self.search(**kwargs, lookahead=False)

    def search(self, q="", filters=None, skip=0, library="main", *, lookahead=True):
        filters = filters or []
        signature = json.dumps([q, filters], sort_keys=True, separators=(",", ":"))
        # Different sections and restarted leaves must never reuse old pages.
        for key in list(self.pending):
            if key[0] == library and (key[1] != signature or skip == 0 or key[2] < skip):
                self._drop(key)
        key = (library, signature, skip)
        kwargs = dict(q=q, filters=filters, skip=skip, library=library)
        request = self.pending.pop(key, None)
        if request is not None:
            self.stats["cache_hits"] += 1
        future, cancel = request if request is not None else self._submit(kwargs)
        try:
            response = future.result()
        except BaseException:
            self.clear()
            raise
        # These are hints only. Catalog.step still chooses the exact next offset
        # and validates schema, identity, order, count, and partition membership.
        count = response.get("count") if isinstance(response, dict) else None
        results = response.get("results") if isinstance(response, dict) else None
        size = len(results) if isinstance(results, list) else 0
        if lookahead and type(count) is int and 0 < size and skip + size < count <= self.max_leaf:
            for offset in (skip + size, skip + 2 * size):
                next_key = (library, signature, offset)
                if offset >= count or next_key in self.pending:
                    continue
                # At most four retained pages across both libraries, with only
                # two network tasks executing under one persisted rate gate.
                if len(self.pending) >= 4:
                    self._drop(next(iter(self.pending)))
                self.pending[next_key] = self._submit({**kwargs, "skip": offset})
        return response
