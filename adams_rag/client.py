from __future__ import annotations

import email.utils
import gzip
import hashlib
import http.client
import io
import json
import random
import threading
import time
import urllib.error
import urllib.request
import zlib


API_URL = "https://adams-api.nrc.gov/aps/api/search"


class BudgetStop(RuntimeError):
    """Normal checkpoint boundary: another invocation can continue later."""


class TemporaryRequestError(RuntimeError):
    """Network or retryable server failures; safe to retry after a cooldown."""


class AdamsHTTPError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"ADAMS returned HTTP {status}; request body and credentials omitted")


class RequestBudget:
    """One bounded request allowance shared by cooperating fetch threads."""
    def __init__(self, remaining):
        self.lock = threading.Lock()
        self._remaining = remaining

    @property
    def remaining(self):
        with self.lock:
            return self._remaining

    @remaining.setter
    def remaining(self, value):
        with self.lock:
            self._remaining = value

    def take(self):
        with self.lock:
            if self._remaining <= 0:
                raise BudgetStop("Per-run API budget reached; rerun the same command to resume")
            self._remaining -= 1


def retry_seconds(value, now):
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            date = email.utils.parsedate_to_datetime(value)
            return max(0, date.timestamp() - now)
        except (ValueError, TypeError, OverflowError):
            return None


class RateGate:
    """Shared persisted request pacing and rolling 24-hour budget per API key."""
    def __init__(self, store, key, rps=0.5, daily_limit=5000, clock=time.time, sleep=time.sleep):
        if rps <= 0 or daily_limit < 1:
            raise ValueError("Rate and daily request budget must be positive")
        self.db = store.db
        self.scope = hashlib.sha256(key.encode()).hexdigest()
        self._interval, self.daily_limit = 1 / rps, daily_limit
        self.clock, self.sleep = clock, sleep
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS http_requests(scope TEXT, at REAL);
            CREATE INDEX IF NOT EXISTS request_window ON http_requests(scope,at);
            CREATE TABLE IF NOT EXISTS http_pacing(scope TEXT PRIMARY KEY, slowed_interval REAL NOT NULL DEFAULT 0);
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO http_state(scope) VALUES(?)", (self.scope,))
            self.db.execute("INSERT OR IGNORE INTO http_pacing(scope) VALUES(?)", (self.scope,))

    @property
    def interval(self):
        slowed = self.db.execute("SELECT slowed_interval FROM http_pacing WHERE scope=?", (self.scope,)).fetchone()[0]
        return max(self._interval, slowed)

    @interval.setter
    def interval(self, value):
        self._interval = value

    def wait(self, reserve=None):
        while True:
            now = self.clock()
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute("DELETE FROM http_requests WHERE scope=? AND at<=?", (self.scope, now - 86400))
                used = self.db.execute("SELECT count(*) FROM http_requests WHERE scope=?", (self.scope,)).fetchone()[0]
                if used >= self.daily_limit:
                    raise BudgetStop("Local rolling 24-hour API budget reached; resume later")
                next_at = self.db.execute("SELECT next_at FROM http_state WHERE scope=?", (self.scope,)).fetchone()[0]
                delay = max(0, next_at - now)
                if delay:
                    self.db.rollback()
                    if delay > 60:
                        raise BudgetStop(f"Server cooldown is saved; resume in at least {int(delay) + 1} seconds")
                    self.sleep(delay)
                    continue
                if reserve is not None:
                    reserve()
                self.db.execute("INSERT INTO http_requests VALUES(?,?)", (self.scope, now))
                self.db.execute("UPDATE http_state SET next_at=? WHERE scope=?", (now + self.interval, self.scope))
                self.db.commit()
                return
            except BaseException:
                self.db.rollback()
                raise

    def defer(self, seconds):
        with self.db:
            self.db.execute("UPDATE http_state SET next_at=max(next_at,?) WHERE scope=?",
                            (self.clock() + seconds, self.scope))

    def slow_down(self):
        """Persist a lower pace across threads and restarts after a throttle."""
        interval = self.interval
        with self.db:
            self.db.execute("UPDATE http_pacing SET slowed_interval=max(slowed_interval,?) WHERE scope=?",
                            (max(interval, min(5.0, interval * 2)), self.scope))


class AdamsClient:
    def __init__(self, store, key, max_requests=20, rps=0.5, daily_limit=5000,
                 opener=urllib.request.urlopen, gate=None, include_search_content=True,
                 budget=None, cancelled=None):
        if not key:
            raise ValueError("Set ADAMS_API_KEY locally after registering at https://adams-api-developer.nrc.gov/")
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        self.key = key
        self.budget = budget or RequestBudget(max_requests)
        self.cancelled = cancelled or (lambda: False)
        self.opener = opener
        self.include_search_content = include_search_content
        self.gate = gate or RateGate(store, key, rps, daily_limit)
        self.db = store.db
        self.scope = hashlib.sha256(key.encode()).hexdigest()
        self.db.execute("""CREATE TABLE IF NOT EXISTS http_responses(
          id INTEGER PRIMARY KEY, scope TEXT NOT NULL, at REAL NOT NULL,
          status INTEGER, outcome TEXT NOT NULL, elapsed_seconds REAL NOT NULL,
          retry_after_seconds REAL, quota_headers TEXT NOT NULL)""")
        self.db.commit()

    @property
    def remaining(self):
        return self.budget.remaining

    @remaining.setter
    def remaining(self, value):
        self.budget.remaining = value

    def reserve(self):
        if self.cancelled():
            raise BudgetStop("Speculative page request cancelled")
        self.budget.take()

    def record_response(self, status, outcome, started, retry_after=None, headers=None):
        # Only response status/timing and known quota headers are retained.
        # No credentials, request bodies, URLs, cookies, or response content.
        allowed = {"retry-after", "ratelimit-limit", "ratelimit-remaining", "ratelimit-reset",
                   "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
                   "x-rate-limit-limit", "x-rate-limit-remaining", "x-rate-limit-reset"}
        quota = {k.lower(): v for k,v in (headers.items() if headers else []) if k.lower() in allowed}
        with self.db:
            self.db.execute("INSERT INTO http_responses(scope,at,status,outcome,elapsed_seconds,retry_after_seconds,quota_headers) VALUES(?,?,?,?,?,?,?)",
                            (self.scope, time.time(), status, outcome, time.monotonic()-started, retry_after, json.dumps(quota)))

    def request(self, payload=None, accession=None):
        from urllib.parse import quote
        url = API_URL + ("/" + quote(accession, safe="") if accession else "")
        data = json.dumps(payload).encode() if payload is not None else None
        for attempt in range(6):
            if self.cancelled():
                raise BudgetStop("Speculative page request cancelled")
            if self.remaining <= 0:
                raise BudgetStop("Per-run API budget reached; rerun the same command to resume")
            self.gate.wait(reserve=self.reserve)
            request = urllib.request.Request(url, data=data, headers={
                "Ocp-Apim-Subscription-Key": self.key,
                "Accept": "application/json", "Content-Type": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": "adams-rag-prototype/0.1",
            })
            started = time.monotonic()
            try:
                with self.opener(request, timeout=60) as response:
                    self.record_response(getattr(response,"status",200), "http_response", started,
                                         headers=getattr(response,"headers",None))
                    raw = response.read(64 * 1024 * 1024 + 1)
                    if len(raw) > 64 * 1024 * 1024:
                        raise RuntimeError("API response exceeds 64 MiB; narrow the query")
                    if getattr(response, "headers", {}).get("Content-Encoding", "").lower() == "gzip":
                        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
                            raw = compressed.read(64 * 1024 * 1024 + 1)
                        if len(raw) > 64 * 1024 * 1024:
                            raise RuntimeError("Decoded API response exceeds 64 MiB; narrow the query")
                    return json.loads(raw)
            except urllib.error.HTTPError as error:
                status = error.code
                wait = retry_seconds(error.headers.get("Retry-After"), time.time())
                self.record_response(status, "http_error", started, wait, error.headers)
                error.close()
                if status not in {408, 429, 500, 502, 503, 504}:
                    raise AdamsHTTPError(status) from None
                if status == 429:
                    self.gate.slow_down()
                self.gate.defer(wait if wait is not None else min(60, 2 ** (attempt + 1)) + random.random())
            except (json.JSONDecodeError, UnicodeDecodeError, gzip.BadGzipFile, EOFError, zlib.error):
                # An HTTP 200 can still contain a cut-off body. Retry the same
                # request; never return a partial page or log response content.
                self.record_response(None, "invalid_response", started)
                if hasattr(self.opener, "close"):
                    self.opener.close()
                self.gate.defer(min(60, 2 ** (attempt + 1)) + random.random())
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                self.record_response(None, "network_error", started)
                if hasattr(self.opener, "close"):
                    self.opener.close()
                self.gate.defer(min(60, 2 ** (attempt + 1)) + random.random())
        raise TemporaryRequestError("ADAMS temporary request failures persisted through retries; existing progress is saved")

    def search(self, q="", filters=None, skip=0, library="main"):
        return self.request({
            "q": q, "content": self.include_search_content, "filters": filters or [], "anyFilters": [],
            "mainLibFilter": library in {"main", "both"},
            "legacyLibFilter": library in {"legacy", "both"},
            "sort": "AccessionNumber", "sortDirection": 0, "skip": skip,
        })


def unpack_search(response):
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise ValueError("Unexpected APS search schema; checkpoint not advanced")
    docs = []
    for item in response["results"]:
        if not isinstance(item, dict) or not isinstance(item.get("document"), dict):
            raise ValueError("Unexpected APS result item; checkpoint not advanced")
        doc = item["document"]
        if not isinstance(doc.get("AccessionNumber"), str) or not doc["AccessionNumber"].strip():
            raise ValueError("APS result lacks accession; checkpoint not advanced")
        docs.append(doc)
    return docs
