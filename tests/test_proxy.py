import http.client
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from adams_rag.server import handler_for


class App:
    def status(self): return {'ready': True}
    def submit(self, payload, generate=True): return {'id': 'test', 'generate': generate}


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(App(), 'https://workspace.example', '/user/test/port/8765/'))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def request(self, path, host='workspace.example', origin=None, method='GET'):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        headers = {'Host': host}
        if origin: headers['Origin'] = origin
        if method == 'POST': headers['Content-Type'] = 'application/json'
        try:
            connection.request(method, path, body='{}' if method == 'POST' else None, headers=headers)
            response = connection.getresponse(); return response.status, response.read()
        finally: connection.close()

    def test_proxy_prefix_and_stripped_paths_both_work(self):
        for path in ['/api/status', '/user/test/port/8765/api/status']:
            code, body = self.request(path)
            self.assertEqual(code, 200); self.assertTrue(json.loads(body)['ready'])
        self.assertEqual(self.request('/user/test/port/8765/')[0], 200)

    def test_only_exact_configured_host_and_origin_are_accepted(self):
        self.assertEqual(self.request('/api/status', host='attacker.example')[0], 403)
        self.assertEqual(self.request('/v1/rag', origin='https://attacker.example', method='POST')[0], 403)
        self.assertEqual(self.request('/user/test/port/8765/v1/rag', origin='https://workspace.example', method='POST')[0], 202)

    def test_invalid_proxy_configuration_is_rejected(self):
        for origin in ['https://user:pass@example.com', 'https://example.com/path', 'http://example.com', 'https://example.com?x=y']:
            with self.assertRaises(ValueError): handler_for(App(), origin)
        with self.assertRaises(ValueError): handler_for(App(), base_path='/../wrong')
