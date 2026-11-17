from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from adams_rag.api_budget import ApiBudget, dollars
from adams_rag.hosted_engine import HostedEngine, load_key
from adams_rag.ollama_engine import make_engine
from adams_rag.research import research
from adams_rag.server import Application, handler_for
from adams_rag.store import Store
from http.server import ThreadingHTTPServer


class HostedEngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.key = self.root / 'test.key'
        self.key.write_text('test-only-not-a-real-key')
        self.key.chmod(0o600)
        self.engine = HostedEngine('openai/gpt-oss-120b', str(self.key), self.root / 'budget.sqlite', 1, enabled=True)
        self.evidence = [{'citation': 1, 'accession': 'DEMO', 'title': 'Test', 'text': 'The pump leaked.'}]

    def response(self, source=1, finish='stop'):
        return {'usage': {'prompt_tokens': 1000, 'completion_tokens': 100},
                'choices': [{'finish_reason': finish, 'message': {'content': json.dumps({
                    'paragraphs': [{'text': 'The pump leaked.', 'source_ids': [source]}],
                    'insufficient_evidence': False, 'gaps': []})}}]}

    def test_disabled_default_cannot_send_and_status_is_free(self):
        engine = HostedEngine('openai/gpt-oss-20b', str(self.key), self.root / 'off.sqlite')
        with patch.object(engine, '_request') as request:
            self.assertEqual(engine.availability()['state'], 'model_unavailable')
            with self.assertRaises(RuntimeError):
                engine.complete('system', {})
            request.assert_not_called()
        self.assertEqual(self.engine.availability()['state'], 'configured')

    def test_budget_reconciles_actual_billed_tokens_and_survives_restart(self):
        with patch.object(self.engine, '_request', return_value=self.response()) as request:
            result = self.engine.generate(None, 'What happened?', self.evidence)
        self.assertEqual(result['answer'], 'The pump leaked. [1]')
        body = json.loads(request.call_args.args[0])
        self.assertNotIn('tools', body)
        self.assertEqual(body['response_format'], {'type': 'json_object'})
        self.assertIn('untrusted source data', body['messages'][0]['content'])
        ledger = ApiBudget(self.root / 'budget.sqlite', 1)
        self.assertEqual(ledger.status()['accounted_usd'], 0.00021)
        self.assertEqual(ledger.status()['uncertain_or_inflight_calls'], 0)

    def test_uncertain_request_keeps_reservation_and_never_retries(self):
        with patch.object(self.engine, '_request', side_effect=RuntimeError('timeout')) as request:
            with self.assertRaises(RuntimeError):
                self.engine.complete('system', {})
            request.assert_called_once()
        self.assertEqual(self.engine.budget.status()['uncertain_or_inflight_calls'], 1)
        self.assertGreater(self.engine.budget.status()['accounted_usd'], 0)

    def test_missing_or_invalid_usage_keeps_reservation(self):
        for usage in (None, {}, {'prompt_tokens': True, 'completion_tokens': 1},
                      {'prompt_tokens': 1, 'completion_tokens': 9000}):
            with self.subTest(usage=usage), patch.object(self.engine, '_request', return_value={'usage': usage}):
                with self.assertRaises(ValueError):
                    self.engine.complete('system', {})
        self.assertEqual(self.engine.budget.status()['uncertain_or_inflight_calls'], 4)

    def test_invalid_citations_and_truncated_output_are_withheld_but_usage_counted(self):
        for response in (self.response(source=2), self.response(finish='length')):
            with patch.object(self.engine, '_request', return_value=response):
                with self.assertRaises(ValueError):
                    self.engine.generate(None, 'question', self.evidence)
        self.assertEqual(self.engine.budget.status()['accounted_usd'], 0.00042)

    def test_key_permissions_and_configuration_fail_closed(self):
        self.key.chmod(0o644)
        with self.assertRaises(ValueError):
            load_key(self.key)
        for model in ('arbitrary', 'https://example.com'):
            with self.assertRaises(ValueError):
                HostedEngine(model, str(self.key), self.root / 'budget.sqlite')
        for amount in ('nan', 'inf', '-1'):
            with self.assertRaises(ValueError):
                dollars(amount)

    def test_https_destination_and_error_redaction(self):
        with patch('adams_rag.hosted_engine.http.client.HTTPSConnection') as connection:
            connection.return_value.getresponse.return_value.status = 401
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401'):
                self.engine._request(b'{}', 'secret')
            connection.assert_called_once_with('api.groq.com', timeout=90)
            connection.return_value.getresponse.return_value.read.assert_not_called()
            connection.return_value.close.assert_called_once()

    def test_factory_requires_explicit_provider_and_rejects_mixed_backends(self):
        with patch.dict('os.environ', {'ADAMS_MODEL_PROVIDER': 'groq', 'ADAMS_MODEL_MONTHLY_USD': '0'}):
            self.assertIsInstance(make_engine(self.root / 'absent', self.root), HostedEngine)
            with self.assertRaises(ValueError):
                make_engine(self.root, self.root, ollama_model='test')

    def test_zero_budget_stops_before_network(self):
        self.engine.budget.limit = 0
        with patch.object(self.engine, '_request') as request:
            with self.assertRaises(RuntimeError):
                self.engine.complete('system', {})
            request.assert_not_called()

    def test_full_pipeline_with_mocked_provider_preserves_source_bindings(self):
        store = Store(self.root / 'research.sqlite')
        self.addCleanup(store.db.close)
        with store.db:
            store.put({'AccessionNumber': 'DEMO', 'DocumentTitle': 'Test', 'content': 'The pump leaked.'})
        with patch.object(self.engine, '_request', return_value=self.response()):
            result = research(store, 'pump leaked', engine=self.engine, live=False)
        self.assertEqual(result['mode'], 'generated')
        self.assertEqual(result['sources'][0]['accession'], 'DEMO')
        self.assertEqual(result['answer'], 'The pump leaked. [1]')

    def test_budget_reservation_is_atomic_across_connections(self):
        ledger = ApiBudget(self.root / 'atomic.sqlite', '0.01')
        def attempt(_):
            try:
                return ledger.reserve('test', dollars('.007'))
            except RuntimeError:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
        self.assertEqual(sum(x is not None for x in results), 1)
        self.assertEqual(ledger.status()['accounted_usd'], .007)

    def test_month_rollover_does_not_erase_uncertain_old_charges(self):
        ledger = self.engine.budget
        with patch.object(ledger, 'month', return_value='2026-11'):
            identifier = ledger.reserve('test', dollars('.5'))
        with patch.object(ledger, 'month', return_value='2026-12'):
            self.assertEqual(ledger.status()['accounted_usd'], 0)
            ledger.settle(identifier, dollars('.3'), 1, 1)
        with patch.object(ledger, 'month', return_value='2026-11'):
            self.assertEqual(ledger.status()['accounted_usd'], .3)

    def test_queue_is_bounded_and_public_serving_can_disable_live_calls(self):
        app = Application(self.root/'db', self.root/'rate', self.root/'catalog',
                          self.root/'missing', self.root/'index', max_queued=1, offline_only=True)
        self.addCleanup(app.pool.shutdown, wait=True)
        with patch.object(app.pool, 'submit') as submit:
            app.submit({'question': 'one', 'live': True})
            app.submit({'question': 'two'})
            self.assertFalse(submit.call_args.args[2]['live'])
            with self.assertRaises(RuntimeError):
                app.submit({'question': 'three'})

    def test_health_endpoint_never_checks_the_paid_model(self):
        app = Application(self.root/'db', self.root/'rate', self.root/'catalog', self.root/'missing', self.root/'index')
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(app, 'status', side_effect=AssertionError('expensive status called')):
                connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
                try:
                    connection.request('GET', '/healthz')
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.loads(response.read()), {'state': 'alive'})
                finally:
                    connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            app.pool.shutdown(wait=True)
