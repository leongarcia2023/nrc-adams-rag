"""Use the source-linked research pipeline with a local Ollama model."""
import http.client
from contextlib import nullcontext
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from .models import MLXEngine, decode_object, model_ready, assemble_answer
from .inference_guard import inference_slot


class OllamaEngine(MLXEngine):
    # Reuse plan/generate/assemble_answer; replace all MLX execution methods.
    def __init__(self, model, base_url="http://127.0.0.1:11434", max_tokens=1400, timeout=300, num_threads=None,
                 storage_path=None, keep_alive=0, cpu_only=False):
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
        if type(keep_alive) is not int or not 0 <= keep_alive <= 600:
            raise ValueError("Keep the local model loaded for between 0 and 600 seconds")
        if type(cpu_only) is not bool:
            raise ValueError("cpu_only must be a boolean")
        self.model = model
        self.host, self.port = parsed.hostname, parsed.port or 11434
        self.max_tokens, self.timeout = max_tokens, timeout
        self.num_threads = num_threads
        self.keep_alive, self.cpu_only = keep_alive, cpu_only
        self.inference_lock = Path(storage_path) / 'models' / 'inference.lock' if storage_path else None
        self.lock = threading.RLock()
        self.progress = None
        self.last_metrics = None

    def _post(self, path, payload, timeout=None):
        # Direct loopback connection: no environment proxies or redirects.
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        deadline = time.monotonic() + (timeout or self.timeout)
        try:
            conn.request("POST", path, body=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"})
            request_socket = conn.sock
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError("Local Ollama request failed; check the server and installed model")
            if payload.get('stream'):
                parts, size, announced = [], 0, False
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError()
                    if request_socket:
                        request_socket.settimeout(remaining)
                    line = response.readline(1024 * 1024 + 1)
                    size += len(line)
                    if not line or size > 4 * 1024 * 1024:
                        raise ValueError('Local model stream ended without a complete answer')
                    if time.monotonic() > deadline:
                        raise TimeoutError()
                    item = json.loads(line)
                    if not isinstance(item, dict) or item.get('error'):
                        raise ValueError('Local model returned an invalid stream')
                    content = item.get('message', {}).get('content', '')
                    if not isinstance(content, str):
                        raise ValueError('Local model returned invalid text')
                    if content:
                        parts.append(content)
                        if self.progress and not announced:
                            self.progress('Drafting cited answer', 'The model has finished reading and is drafting the answer')
                            announced = True
                    if item.get('done'):
                        return {**item, 'message': {'content': ''.join(parts)}}
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Local Ollama response exceeded the size limit")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Local Ollama returned an invalid object")
            return value
        except TimeoutError:
            raise RuntimeError("Local Ollama exceeded its request timeout; the retrieved sources remain available. A faster model service is needed for this request.") from None
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
                "execution": "local_ollama", "resources_managed_by": "ollama",
                "cpu_only": self.cpu_only, "num_threads": self.num_threads,
                "keep_alive_seconds": self.keep_alive, "request_timeout_seconds": self.timeout,
                "context_byte_limit": 3400 if self.cpu_only else None}

    def complete(self, system, data, max_tokens=None):
        tokens = self.max_tokens if max_tokens is None else max_tokens
        if type(tokens) is not int or not 1 <= tokens <= 4096:
            raise ValueError("Use a model token allowance between 1 and 4,096")
        with self.lock, (inference_slot(self.inference_lock) if self.inference_lock else nullcontext()):
            self._check_model()
            if self.progress:
                self.progress('Reading source context', 'The CPU model is reading the selected excerpts; this can take a few minutes')
            options = {"temperature": 0, "num_predict": tokens, "num_ctx": 4096 if self.cpu_only else 16384}
            if self.num_threads is not None:
                options["num_thread"] = self.num_threads
            if self.cpu_only:
                options["num_gpu"] = 0
            response_format = 'json'
            if self.cpu_only and isinstance(data, dict) and data.get('sources'):
                from .cpu_context import answer_schema
                response_format = answer_schema({s['source_id'] for s in data['sources']})
            response = self._post("/api/chat", {
                "model": self.model, "stream": True, "format": response_format, "keep_alive": self.keep_alive,
                "options": options,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
            })
            self.last_metrics = {k: response[k] for k in ('total_duration', 'load_duration', 'prompt_eval_count',
                'prompt_eval_duration', 'eval_count', 'eval_duration') if k in response}
        if response.get("done") is not True or response.get("done_reason") == "length":
            raise ValueError("The local model did not finish its answer; answer withheld")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Local Ollama did not return a text response")
        return decode_object(message["content"])

    def generate(self, model, question, evidence):
        if not self.cpu_only:
            return super().generate(model, question, evidence)
        from .cpu_context import pack_compact
        system, data, context = pack_compact(question, evidence)
        result = self.complete(system, data, max_tokens=min(self.max_tokens, 384))
        answer = assemble_answer(result, len(evidence))
        supplied = {s['source_id'] for s in data['sources']}
        if any(c not in supplied for claim in answer['claims'] for c in claim['source_ids']):
            raise ValueError('The model cited a passage outside its reading budget; answer withheld')
        from .cpu_context import check_compact_answer
        check_compact_answer(answer, data)
        answer['generation_context'] = context
        answer['generation_metrics'] = self.last_metrics
        return answer

    def generate_with_progress(self, question, evidence, progress):
        with self.lock:
            self.progress = progress
            try:
                return self.generate(None, question, evidence)
            finally:
                self.progress = None


def make_engine(model_path, storage_path=None, ollama_model=None):
    provider = os.environ.get("ADAMS_MODEL_PROVIDER", "local")
    if provider != "local":
        if provider != "groq" or ollama_model:
            raise ValueError("Select either the Groq API provider or a local model, not both")
        from .hosted_engine import from_environment
        return from_environment(storage_path or ".")
    if ollama_model:
        threads = os.environ.get("ADAMS_OLLAMA_THREADS")
        cpu_only = os.environ.get("ADAMS_OLLAMA_CPU_ONLY", "0")
        if cpu_only not in {"0", "1"}:
            raise ValueError("ADAMS_OLLAMA_CPU_ONLY must be 0 or 1")
        return OllamaEngine(ollama_model,
                            base_url=os.environ.get("ADAMS_OLLAMA_URL", "http://127.0.0.1:11434"),
                            timeout=min(int(os.environ.get("ADAMS_OLLAMA_TIMEOUT", "300")), 240) if cpu_only == '1'
                                else int(os.environ.get("ADAMS_OLLAMA_TIMEOUT", "300")),
                            num_threads=int(threads) if threads else None,
                            storage_path=storage_path,
                            keep_alive=int(os.environ.get("ADAMS_OLLAMA_KEEP_ALIVE_SECONDS", "0")),
                            cpu_only=cpu_only == "1")
    return MLXEngine(model_path, storage_path=storage_path) if model_ready(model_path) else None
