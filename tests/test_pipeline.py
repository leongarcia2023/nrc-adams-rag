import datetime as dt
import email.utils
import gzip
import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from adams_rag.client import AdamsClient, BudgetStop, RateGate, TemporaryRequestError, retry_seconds, unpack_search
from adams_rag.ingest import archive, crawl, hydrate
from adams_rag.retrieval import Ollama, ask, embed_pending, retrieve, validate_answer
from adams_rag.store import Store, chunk_text, digest


def doc(accession="TEST-A", text="The pump inspection identified a valve issue.", **kwargs):
    return {"AccessionNumber": accession, "DocumentTitle": "Test fixture", "content": text,
            "Url": "https://www.nrc.gov/example.pdf", **kwargs}


def response(*docs):
    return {"results": [{"document": d} for d in docs], "count": len(docs)}


class FakeClient:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.pages)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClock:
    def __init__(self):
        self.now = 1000000.
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeEngine:
    def embed(self, model, texts):
        return [[1., 0.] if "valve" in text else [0., 1.] for text in texts]

    def generate(self, model, question, evidence):
        return {"answer": "The test fixture reports a valve issue [1].", "insufficient_evidence": False}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "test.sqlite")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def put(self, record, **kwargs):
        with self.store.db:
            return self.store.put(record, **kwargs)

    def test_idempotent_import_and_replacement_invalidate_old_evidence(self):
        self.put(doc())
        self.put(doc())
        self.assertEqual(self.store.status()["chunks"], 1)
        embed_pending(self.store, FakeEngine(), "test")
        self.put(doc(text="New evidence about coolant temperature."))
        self.assertFalse(self.store.lexical("valve"))
        self.assertEqual(self.store.status()["vectors_by_model"], {})
        self.assertTrue(self.store.lexical("coolant"))

    def test_metadata_only_refresh_preserves_pdf_pages(self):
        self.put(doc(), provenance="pdf", pages=["Valve evidence on first page", "Second page"])
        self.put(doc(text=""), provenance="api_search_unverified")
        result = retrieve(self.store, "Valve")
        self.assertEqual(result[0]["page"], 1)
        self.assertEqual(result[0]["provenance"], "pdf")
        self.assertTrue(result[0]["url"].endswith("#page=1"))

    def test_pdf_import_upgrades_same_text_to_page_citation(self):
        self.put(doc(text="Valve evidence"), provenance="api_search_unverified")
        self.put(doc(), provenance="pdf", pages=["Valve evidence"])
        self.assertEqual(retrieve(self.store, "valve")[0]["page"], 1)

    def test_title_update_preserves_text_but_invalidates_title_embeddings(self):
        self.put(doc(), provenance="pdf", pages=["Valve evidence"])
        embed_pending(self.store, FakeEngine(), "test")
        self.put(doc(text="", DocumentTitle="New title"), provenance="api_search_unverified")
        self.assertEqual(self.store.status()["vectors_by_model"], {})
        self.assertEqual(retrieve(self.store, "valve")[0]["title"], "New title")

    def test_missing_text_and_packages_do_not_become_evidence(self):
        self.put(doc(text=""))
        self.put(doc("TEST-B", "Metadata text", IsPackage="Yes"))
        self.assertEqual(self.store.status()["documents"], {"missing_text": 1, "package": 1})
        self.assertEqual(self.store.status()["chunks"], 0)

    def test_search_filters_docket_and_does_not_execute_query_syntax(self):
        self.put(doc(DocketNumber=["05000373"]))
        self.put(doc("TEST-B", "Valve at another site", DocketNumber=["05000001"]))
        self.assertEqual({r["accession"] for r in retrieve(self.store, '" OR valve *', docket="05000373")}, {"TEST-A"})
        self.assertFalse(retrieve(self.store, "!!!"))

    def test_page_checkpoint_resume_and_raw_response_archiving(self):
        first = FakeClient([response(doc()), BudgetStop("pause")])
        with self.assertRaises(BudgetStop):
            crawl(self.store, first, "2024-01-23", "2024-01-23")
        self.assertEqual(first.calls[0]["filters"][0]["value"],
                         "(DateAddedTimestamp ge '2024-01-23') and (DateAddedTimestamp le '2024-01-24')")
        self.assertEqual(self.store.db.execute("SELECT offset FROM crawl").fetchone()[0], 1)
        second = FakeClient([response(doc("TEST-B")), response()])
        crawl(self.store, second, "2024-01-23", "2024-01-23")
        self.assertEqual(second.calls[0]["skip"], 1)
        self.assertEqual(self.store.status()["documents"], {"search_excerpt": 2})
        self.assertEqual(self.store.status()["crawl_partitions"], {"exhausted_unverified": 1})
        self.assertFalse(self.store.status()["full_archive_verified"])
        self.assertEqual(len(list((Path(self.tmp.name) / "raw").rglob("*.gz"))), 3)

    def test_raw_archive_shards_new_responses_and_reuses_legacy_files(self):
        payload = response(doc())
        archive(self.store, payload)
        archive(self.store, payload)
        root = Path(self.tmp.name) / "raw"
        files = list(root.rglob("*.gz"))
        self.assertEqual(len(files), 1)
        self.assertEqual(len(files[0].relative_to(root).parts), 3)
        old_payload = response(doc("LEGACY"))
        raw = json.dumps(old_payload, sort_keys=True, ensure_ascii=False)
        legacy = root / (digest(raw) + ".json.gz")
        legacy.write_bytes(gzip.compress(raw.encode()))
        archive(self.store, old_payload)
        self.assertEqual(len(list(root.rglob("*.gz"))), 2)
        self.assertTrue(legacy.exists())

    def test_page_failure_rolls_back_all_documents_and_checkpoint(self):
        invalid = doc("TEST-B", ["not a string"])
        with self.assertRaises(ValueError):
            crawl(self.store, FakeClient([response(doc(), invalid)]), "2024-01-23", "2024-01-23")
        self.assertEqual(self.store.status()["documents"], {})
        self.assertEqual(self.store.db.execute("SELECT offset FROM crawl").fetchone()[0], 0)

    def test_repeated_pages_and_offset_ceilings_are_flagged(self):
        crawl(self.store, FakeClient([response(doc()), response(doc())]), "2024-01-23", "2024-01-23")
        self.assertEqual(self.store.status()["crawl_partitions"], {"blocked": 1})
        crawl(self.store, FakeClient([response(doc())]), "2024-01-24", "2024-01-24", max_offset=1)
        self.assertEqual(self.store.status()["crawl_partitions"], {"blocked": 2})

    def test_count_disagrees_with_empty_page(self):
        crawl(self.store, FakeClient([{"results": [], "count": 50}]), "2024-01-23", "2024-01-23")
        self.assertEqual(self.store.status()["crawl_partitions"], {"blocked": 1})

    def test_schema_drift_never_looks_like_no_results(self):
        for value in ({"value": []}, {"results": [{}]}, {"results": [{"document": {}}]}):
            with self.assertRaises(ValueError):
                unpack_search(value)

    def test_missing_text_hydration_records_no_text_without_retry_loop(self):
        self.put(doc(text=""))
        class Client:
            def request(self, **kwargs):
                return {"document": doc(text="")}
        hydrate(self.store, Client())
        self.assertEqual(self.store.status()["documents"], {"unavailable_text": 1})
        hydrate(self.store, Client())
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM retrieval_events").fetchone()[0], 1)

    def test_search_excerpt_is_hydrated_before_use_as_evidence(self):
        self.put(doc(text="Valve excerpt"), provenance="api_search_excerpt")
        self.assertEqual(self.store.status()["documents"], {"search_excerpt": 1})
        self.assertFalse(self.store.lexical("valve"))
        class Client:
            def request(self, **kwargs):
                return {"document": doc(text="Full document describes valve inspection and corrective action.")}
        hydrate(self.store, Client())
        self.assertTrue(self.store.lexical("corrective action"))
        self.put(doc(text="Changed valve excerpt"), provenance="api_search_excerpt")
        self.assertTrue(self.store.lexical("corrective action"))
        self.assertEqual(self.store.status()["documents"], {"indexed": 1})

    def test_legacy_search_snippets_are_removed_from_evidence_on_open(self):
        self.put(doc())
        with self.store.db:
            self.store.db.execute("UPDATE documents SET provenance='api_search_unverified'")
        self.store.db.close()
        self.store = Store(Path(self.tmp.name) / "test.sqlite")
        self.assertFalse(self.store.lexical("valve"))
        self.assertEqual(self.store.status()["documents"], {"search_excerpt": 1})

    def test_old_date_equality_checkpoint_does_not_skip_range_scan(self):
        old_job=json.dumps({"library":"main", "docket":None}, sort_keys=True)
        with self.store.db:
            self.store.db.execute("INSERT INTO crawl(job,day,state) VALUES(?,?,'exhausted_unverified')", (old_job,'2024-01-23'))
        self.store.db.close()
        self.store = Store(Path(self.tmp.name) / "test.sqlite")
        client = FakeClient([response(doc()), response()])
        crawl(self.store, client, '2024-01-23', '2024-01-23')
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(self.store.status()["crawl_partitions"], {"obsolete_filter":1, "exhausted_unverified":1})

    def test_rate_gate_persists_pacing_budget_and_retry_after(self):
        clock = FakeClock()
        gate = RateGate(self.store, "fake-test-key", rps=1, daily_limit=2, clock=clock.clock, sleep=clock.sleep)
        gate.wait()
        restarted = RateGate(self.store, "fake-test-key", rps=1, daily_limit=2, clock=clock.clock, sleep=clock.sleep)
        restarted.wait()
        self.assertEqual(clock.sleeps, [1])
        with self.assertRaises(BudgetStop):
            restarted.wait()
        clock.now += 86401
        restarted.defer(120)
        with self.assertRaises(BudgetStop):
            restarted.wait()

    def test_retry_after_seconds_and_http_date(self):
        now = 1000000
        self.assertEqual(retry_seconds("30", now), 30)
        date = email.utils.format_datetime(dt.datetime.fromtimestamp(now + 90, dt.timezone.utc))
        self.assertEqual(retry_seconds(date, now), 90)
        self.assertIsNone(retry_seconds("unknown", now))

    def test_429_is_respected_and_retries_count_against_budget(self):
        clock = FakeClock()
        gate = RateGate(self.store, "fake-key", rps=1, clock=clock.clock, sleep=clock.sleep)
        calls = []
        def opener(request, **kwargs):
            calls.append(request)
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "Busy", {"Retry-After": "5"}, io.BytesIO())
            return io.BytesIO(json.dumps(response()).encode())
        client = AdamsClient(self.store, "fake-key", max_requests=2, gate=gate, opener=opener)
        self.assertEqual(client.search()["results"], [])
        self.assertEqual(clock.sleeps, [5])
        logs = self.store.db.execute("SELECT status,retry_after_seconds,quota_headers FROM http_responses ORDER BY id").fetchall()
        self.assertEqual([r["status"] for r in logs], [429,200])
        self.assertEqual(logs[0]["retry_after_seconds"],5)
        self.assertEqual(json.loads(logs[0]["quota_headers"]),{"retry-after":"5"})
        with self.assertRaises(BudgetStop):
            client.search()

    def test_response_log_excludes_sensitive_headers(self):
        clock = FakeClock()
        gate = RateGate(self.store,"fake-key",clock=clock.clock,sleep=clock.sleep)
        def opener(request,**kwargs):
            result=io.BytesIO(json.dumps(response()).encode())
            result.status=200
            result.headers={"X-RateLimit-Remaining":"99","Set-Cookie":"private-cookie","Ocp-Apim-Subscription-Key":"private-key"}
            return result
        AdamsClient(self.store,"fake-key",gate=gate,opener=opener).search()
        row=self.store.db.execute("SELECT * FROM http_responses").fetchone()
        self.assertEqual(json.loads(row["quota_headers"]),{"x-ratelimit-remaining":"99"})
        self.assertNotIn("private",str(dict(row)))

    def test_auth_failure_not_retried_and_key_not_in_error(self):
        clock = FakeClock()
        gate = RateGate(self.store, "fake-key", clock=clock.clock, sleep=clock.sleep)
        def opener(request, **kwargs):
            raise urllib.error.HTTPError(request.full_url, 401, "fake-key", {}, io.BytesIO())
        client = AdamsClient(self.store, "fake-key", gate=gate, opener=opener)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as error:
            client.search()
        self.assertNotIn("fake-key", str(error.exception))

    def test_network_retry_exhaustion_remains_recoverable(self):
        clock=FakeClock()
        gate=RateGate(self.store,"fake-key",clock=clock.clock,sleep=clock.sleep)
        def offline(request,**kwargs):
            raise urllib.error.URLError("temporary offline condition")
        client=AdamsClient(self.store,"fake-key",max_requests=10,gate=gate,opener=offline)
        with self.assertRaises(TemporaryRequestError):
            client.search()
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM http_responses WHERE outcome='network_error'").fetchone()[0],6)
        # A later attempt can continue through the same limiter and request ledger.
        clock.now += 61
        client.opener=lambda request,**kwargs:io.BytesIO(json.dumps(response()).encode())
        self.assertEqual(client.search()["results"],[])

    def test_invalid_response_retries_same_page_without_logging_body(self):
        expected = response(doc())
        for index, body in enumerate((b'{"private-body":"cut off', b'\xff')):
            with self.subTest(body=body):
                clock = FakeClock()
                key = f"fake-key-{index}"
                gate = RateGate(self.store, key, clock=clock.clock, sleep=clock.sleep)
                calls = []
                def opener(request, **kwargs):
                    calls.append(request.data)
                    return io.BytesIO(body if len(calls) == 1 else json.dumps(expected).encode())
                client = AdamsClient(self.store, key, max_requests=2, gate=gate, opener=opener)
                self.assertEqual(client.search(skip=100), expected)
                self.assertEqual(calls[0], calls[1])
                self.assertTrue(clock.sleeps)
                logs = self.store.db.execute("SELECT * FROM http_responses WHERE scope=?", (client.scope,)).fetchall()
                self.assertEqual(sum(row["outcome"] == "invalid_response" for row in logs), 1)
                self.assertNotIn("private-body", str([dict(row) for row in logs]))
                self.assertEqual(client.remaining, 0)

    def test_interrupted_response_body_is_retried(self):
        for index, error in enumerate((http.client.IncompleteRead(b'private-body', 100), ConnectionResetError())):
            with self.subTest(error=type(error).__name__):
                clock = FakeClock()
                key = f"fake-key-{index}"
                gate = RateGate(self.store, key, clock=clock.clock, sleep=clock.sleep)
                class BrokenBody(io.BytesIO):
                    def read(self, *args):
                        raise error
                bodies = iter([BrokenBody(), io.BytesIO(json.dumps(response()).encode())])
                client = AdamsClient(self.store, key, gate=gate, opener=lambda *args, **kwargs: next(bodies))
                self.assertEqual(client.search()["results"], [])
                self.assertEqual(self.store.db.execute("SELECT count(*) FROM http_responses WHERE scope=? AND outcome='network_error'", (client.scope,)).fetchone()[0], 1)

    def test_invalid_response_exhaustion_remains_recoverable_and_budgeted(self):
        clock = FakeClock()
        gate = RateGate(self.store, "fake-key", clock=clock.clock, sleep=clock.sleep)
        client = AdamsClient(self.store, "fake-key", max_requests=10, gate=gate,
                             opener=lambda *args, **kwargs: io.BytesIO(b'{"cut off'))
        with self.assertRaises(TemporaryRequestError):
            client.search(skip=100)
        self.assertEqual(client.remaining, 4)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM http_responses WHERE outcome='invalid_response'").fetchone()[0], 6)
        clock.now += 61
        client.opener = lambda *args, **kwargs: io.BytesIO(json.dumps(response()).encode())
        self.assertEqual(client.search(skip=100)["results"], [])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM http_requests").fetchone()[0], 7)

    def test_gzip_response_and_metadata_only_search(self):
        clock = FakeClock()
        gate = RateGate(self.store, "fake-key", clock=clock.clock, sleep=clock.sleep)
        def opener(request, **kwargs):
            self.assertFalse(json.loads(request.data)["content"])
            self.assertEqual(request.get_header("Accept-encoding"), "gzip")
            body = io.BytesIO(gzip.compress(json.dumps(response(doc())).encode()))
            body.headers = {"Content-Encoding": "gzip"}
            return body
        client = AdamsClient(self.store, "fake-key", gate=gate, opener=opener, include_search_content=False)
        self.assertEqual(client.search(), response(doc()))

    def test_corrupt_gzip_retries_without_returning_a_partial_page(self):
        clock = FakeClock()
        gate = RateGate(self.store, "fake-key", clock=clock.clock, sleep=clock.sleep)
        calls = []
        def opener(request, **kwargs):
            calls.append(1)
            raw = gzip.compress(json.dumps(response()).encode())
            body = io.BytesIO(raw[:-5] if len(calls) == 1 else raw)
            body.headers = {"Content-Encoding": "gzip"}
            return body
        client = AdamsClient(self.store, "fake-key", gate=gate, opener=opener)
        self.assertEqual(client.search(), response())
        self.assertEqual(len(calls), 2)

    def test_throttle_reduces_pace_and_never_speeds_up_a_slow_gate(self):
        clock = FakeClock()
        gate = RateGate(self.store, "fake-key", rps=4, clock=clock.clock, sleep=clock.sleep)
        calls = []
        def opener(request, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "Busy", {"Retry-After": "10"}, None)
            return io.BytesIO(json.dumps(response()).encode())
        self.assertEqual(AdamsClient(self.store, "fake-key", gate=gate, opener=opener).search(), response())
        self.assertEqual(gate.interval, .5)
        self.assertEqual(clock.sleeps, [10])
        gate.interval = 10
        gate.slow_down()
        self.assertEqual(gate.interval, 10)

    def test_hybrid_retrieval_and_generated_answer_path(self):
        self.put(doc())
        self.put(doc("TEST-B", "Weather report about rainfall"))
        engine = FakeEngine()
        self.assertEqual(embed_pending(self.store, engine, "test")["embedded"], 2)
        self.assertEqual(embed_pending(self.store, engine, "test")["embedded"], 0)
        answer = ask(self.store, "valve", "test-chat", "test", engine)
        self.assertEqual(answer["mode"], "generated")
        self.assertEqual(answer["sources"][0]["accession"], "TEST-A")
        validate_answer(answer, len(answer["sources"]))

    def test_invalid_citations_and_uncited_claims_are_rejected(self):
        for answer in ("A claim [9].", "Uncited claim"):
            with self.assertRaises(ValueError):
                validate_answer({"answer": answer, "insufficient_evidence": False}, 2)

    def test_ollama_chat_payload_and_response_validation(self):
        engine = Ollama()
        calls = []
        def post(endpoint, payload):
            calls.append((endpoint, payload))
            return {"message": {"content": json.dumps({"answer": "Reported valve issue [1].", "insufficient_evidence": False})}}
        engine.post = post
        self.assertFalse(engine.generate("test-chat", "Question?", [{"citation": 1, "text": "Valve issue"}])["insufficient_evidence"])
        self.assertEqual(calls[0][0], "/api/chat")
        self.assertFalse(calls[0][1]["stream"])
        self.assertIn("untrusted", calls[0][1]["messages"][0]["content"])
        engine.post = lambda *args: {"message": {"content": '{"answer":"Claim [8]", "insufficient_evidence":false}'}}
        with self.assertRaises(ValueError):
            engine.generate("test-chat", "Question?", [{"citation": 1}])

    def test_ollama_embedding_payload_and_validation(self):
        engine = Ollama()
        calls = []
        def post(endpoint, payload):
            calls.append((endpoint, payload))
            return {"embeddings": [[1., 0.], [0., 1.]]}
        engine.post = post
        self.assertEqual(len(engine.embed("test", ["one", "two"])), 2)
        self.assertEqual(calls[0][0], "/api/embed")
        self.assertFalse(calls[0][1]["truncate"])
        with self.assertRaises(ValueError):
            engine.embed("test", ["one"])
        with self.assertRaises(ValueError):
            Ollama("https://example.com")

    def test_no_model_returns_evidence_and_no_matches_abstains(self):
        self.put(doc())
        answer = ask(self.store, "valve")
        self.assertEqual(answer["mode"], "evidence_only")
        self.assertIsNone(answer["answer"])
        self.assertTrue(ask(self.store, "xylophone")["insufficient_evidence"])

    def test_chunk_overlap_has_no_gaps_or_tail_loop(self):
        text = " ".join(str(i) for i in range(1001))
        chunks = list(chunk_text(text, size=100, overlap=20))
        self.assertEqual(set(" ".join(chunks).split()), set(text.split()))
        self.assertEqual(chunks[0].split()[-20:], chunks[1].split()[:20])
        with self.assertRaises(ValueError):
            list(chunk_text(text, size=100, overlap=100))


if __name__ == "__main__":
    unittest.main()
