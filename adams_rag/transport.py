"""One reusable, certificate-verified connection to the official API."""
import http.client
import urllib.error
from urllib.parse import urlsplit


class PersistentOpener:
    def __init__(self, factory=http.client.HTTPSConnection):
        self.factory = factory
        self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __call__(self, request, timeout=60):
        url = urlsplit(request.full_url)
        if (url.scheme != "https" or url.hostname != "adams-api.nrc.gov"
                or url.port not in (None, 443) or url.username or url.password):
            raise ValueError("Persistent transport accepts only the official HTTPS API origin")
        if self.connection is None:
            self.connection = self.factory(url.hostname, timeout=timeout)
        try:
            self.connection.request(request.get_method(), url.path + ("?" + url.query if url.query else ""),
                                    body=request.data, headers=dict(request.header_items()))
            response = self.connection.getresponse()
        except (OSError, http.client.HTTPException):
            self.close()
            raise
        if not 200 <= response.status < 300:
            # Never redirect a subscription key to another origin, and discard
            # unread error bodies before reusing the connection.
            status, reason, headers = response.status, response.reason, response.headers
            response.close()
            self.close()
            raise urllib.error.HTTPError(request.full_url, status, reason, headers, None)
        return response
