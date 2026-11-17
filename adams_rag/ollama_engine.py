"""Use the source-linked research pipeline with a local Ollama model."""
import http.client
import json
import os
import threading
from urllib.parse import urlsplit

from .models import MLXEngine, decode_object, model_ready


class OllamaEngine(MLXEngine):
    # Reuse plan/generate/assemble_answer; replace all MLX execution methods.
    def __init__(self, model, base_url="http://127.0.0.1:11434", max_tokens=1400, timeout=300, num_threads=None):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("Ollama must run on a local HTTP loopback address")
        if not isinstance(model, str) or not model.strip() or len(model) > 256 or any(c.isspace() for c in model):
            raise ValueError("Specify an installed local Ollama model name")
        if "cloud" in model.lower():
            raise ValueError("Use a local model, not an Ollama cloud model")
        if type(timeout) is not int or not 30 <= timeout <= 1800:
            raise ValueError("Use an Ollama timeout between 30 and 1,800 seconds")
        if num_threads is not None and (type(num_threads) is not int or not 1 <= num_threads <= 256):
            raise ValueError("Use between 1 and 256 Ollama CPU threads")
        self.model = model
        self.host, self.port = parsed.hostname, parsed.port or 11434
        self.max_tokens, self.timeout = max_tokens, timeout
        self.num_threads = num_threads
        self.lock = threading.RLock()

    def _post(self, path, payload, timeout=None):
        # Direct loopback connection: no environment proxies or redirects.
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            conn.request("POST", path, body=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError("Local Ollama request failed; check the server and installed model")
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Local Ollama response exceeded the size limit")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Local Ollama returned an invalid object")
            return value
        except TimeoutError:
            raise RuntimeError("Local Ollama exceeded its request timeout; reduce model size or increase ADAMS_OLLAMA_TIMEOUT") from None
        except (OSError, http.client.HTTPException):
            raise RuntimeError("Cannot reach local Ollama; start it and install the requested model") from None
        finally:
            conn.close()

    def _check_model(self, timeout=None):
        info = self._post("/api/show", {"model": self.model}, timeout=timeout)
        if info.get("remote_host") or info.get("remote_model"):
            raise RuntimeError("This adapter requires locally served weights; remote Ollama models are disabled")

    def availability(self):
        try:
            self._check_model(timeout=3)
        except (RuntimeError, ValueError):
            return {"state": "model_unavailable", "backend": "ollama", "model": self.model}
        return {"state": "ready", "backend": "ollama", "model": self.model,
                "execution": "local_ollama", "resources_managed_by": "ollama"}

    def complete(self, system, data, max_tokens=None):
        tokens = self.max_tokens if max_tokens is None else max_tokens
        if type(tokens) is not int or not 1 <= tokens <= 4096:
            raise ValueError("Use a model token allowance between 1 and 4,096")
        with self.lock:
            self._check_model()
            options = {"temperature": 0, "num_predict": tokens, "num_ctx": 16384}
            if self.num_threads is not None:
                options["num_thread"] = self.num_threads
            response = self._post("/api/chat", {
                "model": self.model, "stream": False, "format": "json", "keep_alive": 0,
                "options": options,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(data)}],
            })
        if response.get("done") is not True or response.get("done_reason") == "length":
            raise ValueError("The local model did not finish its answer; answer withheld")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Local Ollama did not return a text response")
        return decode_object(message["content"])


def make_engine(model_path, storage_path=None, ollama_model=None):
    if ollama_model:
        threads = os.environ.get("ADAMS_OLLAMA_THREADS")
        return OllamaEngine(ollama_model,
                            base_url=os.environ.get("ADAMS_OLLAMA_URL", "http://127.0.0.1:11434"),
                            timeout=int(os.environ.get("ADAMS_OLLAMA_TIMEOUT", "300")),
                            num_threads=int(threads) if threads else None)
    return MLXEngine(model_path, storage_path=storage_path) if model_ready(model_path) else None
