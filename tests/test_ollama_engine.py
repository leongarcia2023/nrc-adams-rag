import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from adams_rag.cli import parser
from adams_rag.ollama_engine import OllamaEngine, make_engine
from adams_rag.research import research
from adams_rag.server import Application
from adams_rag.store import Store


class OllamaEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = OllamaEngine('test-local-model')
        self.evidence = [{'citation': 1, 'accession': 'DEMO', 'title': 'Test', 'text': 'The pump leaked.'}]

    def response(self, **changes):
        return {'done': True, 'done_reason': 'stop', 'message': {'content': json.dumps({
            'paragraphs': [{'text': 'The pump leaked.', 'source_ids': [1]}],
            'insufficient_evidence': False, 'gaps': []})}, **changes}

    def test_cited_generation_uses_the_shared_validation_and_prompt(self):
        with patch.object(self.engine, '_post', side_effect=[{}, self.response()]) as post:
            answer = self.engine.generate(None, 'What happened?', self.evidence)
        self.assertEqual(answer['answer'], 'The pump leaked. [1]')
        payload = post.call_args.args[1]
        self.assertEqual(payload['model'], 'test-local-model')
        self.assertEqual(payload['format'], 'json')
        self.assertFalse(payload['stream'])
        self.assertIn('untrusted source data', payload['messages'][0]['content'])
        self.assertIn('DEMO', payload['messages'][1]['content'])

    def test_invalid_citation_and_incomplete_generation_are_withheld(self):
        invalid = self.response()
        invalid['message']['content'] = json.dumps({'paragraphs': [{'text': 'Claim', 'source_ids': [2]}],
            'insufficient_evidence': False, 'gaps': []})
        for response in [invalid, self.response(done=False), self.response(done_reason='length')]:
            with patch.object(self.engine, '_post', side_effect=[{}, response]):
                with self.assertRaises(ValueError):
                    self.engine.generate(None, 'question', self.evidence)

    def test_cloud_models_and_nonlocal_destinations_are_rejected(self):
        for url in ['https://example.com', 'http://127.0.0.1@example.com',
                    'http://127.0.0.1/path', 'http://127.0.0.1?target=x']:
            with self.assertRaises(ValueError):
                OllamaEngine('test', base_url=url)
        with self.assertRaises(ValueError):
            OllamaEngine('test-cloud')
        with patch.object(self.engine, '_post', return_value={'remote_host': 'https://example.com'}) as post:
            with self.assertRaises(RuntimeError):
                self.engine.generate(None, 'question', self.evidence)
        self.assertEqual(post.call_count, 1)

    def test_availability_and_invalid_token_budget(self):
        with patch.object(self.engine, '_post', side_effect=RuntimeError('unavailable')):
            self.assertEqual(self.engine.availability()['state'], 'model_unavailable')
        with patch.object(self.engine, '_post', return_value={}):
            self.assertEqual(self.engine.availability()['state'], 'ready')
        for tokens in (0, True, 4097):
            with self.assertRaises(ValueError):
                self.engine.complete('system', {}, max_tokens=tokens)

    def test_cli_and_server_select_ollama_without_mlx_weights(self):
        args = parser().parse_args(['research', 'question', '--ollama-model', 'test'])
        self.assertEqual(args.ollama_model, 'test')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(make_engine(root / 'absent'))
            app = Application(root / 'db', root / 'rate', root / 'catalog', root / 'absent', root / 'index', 'test')
            try:
                self.assertIsInstance(app.engine, OllamaEngine)
                with patch.object(app.engine, '_post', return_value={}):
                    self.assertTrue(app.status()['local_model_installed'])
            finally:
                app.pool.shutdown(wait=True)

    def test_loopback_http_and_full_research_pipeline(self):
        response = self.response()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                body = json.dumps({} if self.path == '/api/show' else response).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            engine = OllamaEngine('test', f'http://127.0.0.1:{server.server_port}')
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory) / 'research.sqlite')
                try:
                    with store.db:
                        store.put({'AccessionNumber': 'DEMO', 'DocumentTitle': 'Test', 'content': 'The pump leaked.'})
                    result = research(store, 'pump leaked', engine=engine, live=False)
                    self.assertEqual(result['mode'], 'generated')
                    self.assertEqual(result['answer'], 'The pump leaked. [1]')
                    self.assertEqual(result['sources'][0]['accession'], 'DEMO')
                finally:
                    store.db.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
