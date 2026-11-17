"""Explicitly enabled Groq API adapter; no remote inference in local defaults."""
import http.client
import json
import os
from pathlib import Path
import stat

from .api_budget import ApiBudget
from .models import MLXEngine, decode_object

# Standard published USD / million tokens, verified 2026-11-17.
# Integers below are billionths of a dollar per token. No tools or batch pricing.
MODELS = {"openai/gpt-oss-20b": (75, 300), "openai/gpt-oss-120b": (150, 600)}
CONTEXT_LIMIT = 131072
MAX_RESPONSE = 4 * 1024 * 1024


def load_key(filename):
    path = Path(filename).expanduser()
    if not path.is_absolute():
        raise ValueError("The model API key file must have an absolute path")
    try:
        with path.open() as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or (os.name == "posix" and
                    (info.st_mode & 0o077 or info.st_uid != os.getuid())):
                raise ValueError("The model API key must be an owner-only regular file owned by this process user")
            key = stream.read(4097).strip()
    except OSError:
        raise ValueError("Cannot read the model API key file") from None
    if not key or len(key) > 4096 or not key.isascii() or any(c.isspace() for c in key):
        raise ValueError("The model API key file is empty or malformed")
    return key


class HostedEngine(MLXEngine):
    def __init__(self, model, key_file, budget_path, monthly_usd=0, enabled=False, timeout=90, max_tokens=4096):
        if model not in MODELS:
            raise ValueError("Choose a supported, explicitly priced Groq model")
        if type(timeout) is not int or not 10 <= timeout <= 180:
            raise ValueError("Use a model API timeout between 10 and 180 seconds")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("Use a model token allowance between 1 and 4,096")
        self.model, self.key_file, self.enabled = model, key_file, enabled
        self.timeout, self.max_tokens = timeout, max_tokens
        self.budget = ApiBudget(budget_path, monthly_usd)

    def availability(self):
        # A status check never calls a paid endpoint or sends a key.
        try:
            if not self.enabled or not self.budget.limit:
                raise ValueError
            load_key(self.key_file)
        except (ValueError, TypeError):
            state = "model_unavailable"
        else:
            state = "configured"  # Credentials/quality have not been validated online.
        return {"state": state, "backend": "groq", "model": self.model,
                "execution": "hosted_api", "network_verified": False,
                "budget": self.budget.status()}

    def _request(self, body, key):
        # Fixed HTTPS destination, certificate verification, no redirects or proxy env.
        connection = http.client.HTTPSConnection("api.groq.com", timeout=self.timeout)
        try:
            connection.request("POST", "/openai/v1/chat/completions", body=body,
                               headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                # Do not expose provider error bodies (may echo credentials or prompts).
                raise RuntimeError(f"Model API returned HTTP {response.status}; no automatic retry was made")
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise ValueError("Model API response exceeded the size limit")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("Model API returned an invalid response")
            return result
        except (OSError, http.client.HTTPException):
            raise RuntimeError("Model API connection failed or timed out; the uncertain cost remains reserved") from None
        except (json.JSONDecodeError, UnicodeError):
            raise ValueError("Model API returned invalid JSON") from None
        finally:
            connection.close()

    def complete(self, system, data, max_tokens=None):
        if not self.enabled:
            raise RuntimeError("Hosted inference is disabled; enable it only after approving the provider and budget")
        tokens = self.max_tokens if max_tokens is None else max_tokens
        if type(tokens) is not int or not 1 <= tokens <= 4096:
            raise ValueError("Use a model token allowance between 1 and 4,096")
        key = load_key(self.key_file)
        body = json.dumps({"model": self.model, "stream": False, "temperature": 0,
                           "max_completion_tokens": tokens, "reasoning_effort": "low",
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": json.dumps(data)}]}).encode()
        if len(body) > 240000:
            raise ValueError("Retrieved context exceeds the hosted request size limit")
        input_rate, output_rate = MODELS[self.model]
        # Reserve the full accepted context, not an optimistic token estimate.
        # Timeouts/crashes keep this reservation; only a valid usage receipt lowers it.
        call_id = self.budget.reserve(self.model, CONTEXT_LIMIT * input_rate + tokens * output_rate)
        result = self._request(body, key)
        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("Model API omitted usage; cost remains reserved and answer is withheld")
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if (type(prompt) is not int or type(completion) is not int or
                not 0 <= prompt <= CONTEXT_LIMIT or not 0 <= completion <= tokens):
            raise ValueError("Model API usage is invalid; cost remains reserved and answer is withheld")
        self.budget.settle(call_id, prompt * input_rate + completion * output_rate, prompt, completion)
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("Model API returned no single answer")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("The model did not finish its answer; answer withheld")
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str) or message.get("refusal"):
            raise ValueError("Model API returned no usable answer")
        return decode_object(message["content"])


def from_environment(storage_path):
    return HostedEngine(
        model=os.environ.get("ADAMS_API_MODEL", "openai/gpt-oss-120b"),
        key_file=os.environ.get("ADAMS_MODEL_API_KEY_FILE", ""),
        budget_path=os.environ.get("ADAMS_MODEL_BUDGET_DB", str(Path(storage_path) / "model-budget.sqlite")),
        monthly_usd=os.environ.get("ADAMS_MODEL_MONTHLY_USD", "0"),
        enabled=os.environ.get("ADAMS_ALLOW_HOSTED_MODEL") == "1")
