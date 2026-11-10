import io
import unittest
import urllib.error
import urllib.request

from adams_rag.transport import PersistentOpener


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.connections = []
        self.responses = []
        owner = self
        class Connection:
            def __init__(self, host, timeout):
                owner.connections.append(self)
                self.host, self.requests, self.closed = host, [], False
            def request(self, method, path, **kwargs):
                self.requests.append((method, path, kwargs))
            def getresponse(self):
                value = owner.responses.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value
            def close(self):
                self.closed = True
        self.opener = PersistentOpener(factory=Connection)
        self.request = urllib.request.Request('https://adams-api.nrc.gov/aps/api/search', data=b'{}')

    def response(self, status=200):
        body = io.BytesIO(b'{}')
        body.status, body.reason, body.headers = status, 'Fixture', {'Retry-After': '10'}
        return body

    def test_connection_is_reused_for_successful_requests(self):
        self.responses = [self.response(), self.response()]
        for _ in range(2):
            with self.opener(self.request) as response:
                self.assertEqual(response.read(), b'{}')
        self.assertEqual(len(self.connections), 1)
        self.assertEqual(len(self.connections[0].requests), 2)
        self.opener.close()
        self.assertTrue(self.connections[0].closed)

    def test_disconnect_reopens_and_http_error_preserves_retry_header(self):
        self.responses = [ConnectionResetError(), self.response(429), self.response()]
        with self.assertRaises(ConnectionResetError):
            self.opener(self.request)
        self.assertTrue(self.connections[0].closed)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.opener(self.request)
        self.assertEqual(error.exception.headers['Retry-After'], '10')
        self.assertTrue(self.connections[1].closed)
        with self.opener(self.request) as response:
            self.assertEqual(response.read(), b'{}')
        self.assertEqual(len(self.connections), 3)

    def test_foreign_origins_and_redirects_never_receive_credentials(self):
        for url in ['http://adams-api.nrc.gov/aps', 'https://example.test/aps', 'https://adams-api.nrc.gov:444/aps']:
            with self.assertRaises(ValueError):
                self.opener(urllib.request.Request(url))
        self.assertEqual(self.connections, [])
        redirect = self.response(302)
        redirect.headers['Location'] = 'https://example.test/aps'
        self.responses = [redirect]
        with self.assertRaises(urllib.error.HTTPError):
            self.opener(self.request)
        self.assertEqual(len(self.connections), 1)


if __name__ == '__main__':
    unittest.main()
