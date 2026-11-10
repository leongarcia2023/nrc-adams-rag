"""Exact text windows for a bounded tokenizer; shared by the staged window index."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .mlx_embeddings import FILES, MODEL

POLICY = "exact-input-windows-512-overlap-64-v1"


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Window:
    ordinal: int
    start_character: int
    end_character: int
    input_tokens: int
    text: str


def plan_windows(text, tokenizer, max_tokens=512, overlap_tokens=64):
    if not isinstance(text, str):
        raise ValueError("Use text input")
    if (type(max_tokens) is not int or max_tokens < 4
            or type(overlap_tokens) is not int or not 0 <= overlap_tokens < max_tokens - 2):
        raise ValueError("Use a token limit and a smaller nonnegative overlap")
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Window planning requires untruncated, unpadded tokenization")
    encoded = tokenizer.encode(text)
    if len(encoded.ids) <= max_tokens:
        return [Window(1, 0, len(text), len(encoded.ids), text)]
    offsets = [(a, b) for a, b in encoded.offsets if b > a]
    if any(not 0 <= a < b <= len(text) for a, b in offsets):
        raise ValueError("Invalid tokenizer character offsets")
    special_tokens = sum(a == b for a, b in encoded.offsets)
    allowance = max_tokens - special_tokens
    if allowance <= overlap_tokens or not offsets:
        raise ValueError("Tokenizer leaves no room for window content")
    windows, start_index = [], 0
    while start_index < len(offsets):
        start = 0 if not windows else min(offsets[start_index][0], windows[-1].end_character)
        end_index = min(start_index + allowance, len(offsets))
        while end_index > start_index:
            end = len(text) if end_index == len(offsets) else offsets[end_index - 1][1]
            fragment = text[start:end]
            # Cutting at a subword boundary can change tokenization. Measure the
            # actual string that will be sent to the existing embedding worker.
            count = len(tokenizer.encode(fragment).ids)
            if count <= max_tokens:
                break
            end_index -= 1
        else:
            raise ValueError("A tokenizer span cannot fit the configured allowance")
        if windows and (start > windows[-1].end_character or end <= windows[-1].end_character):
            raise ValueError("Token offsets cannot establish advancing, gap-free windows")
        windows.append(Window(len(windows) + 1, start, end, count, fragment))
        if end == len(text):
            break
        start_index = max(start_index + 1, end_index - overlap_tokens)
    if windows[0].start_character != 0 or windows[-1].end_character != len(text):
        raise ValueError("Window coverage is incomplete")
    return windows


@dataclass(frozen=True)
class PlannedWindow:
    ordinal: int
    start_character: int
    end_character: int
    input_tokens: int
    text: str
    input_hash: str
    token_hash: str
    base_compatible: bool

    def manifest(self):
        return {name: getattr(self, name) for name in ("ordinal", "start_character", "end_character",
            "input_tokens", "input_hash", "token_hash", "base_compatible")}


@dataclass(frozen=True)
class WindowPlan:
    source_hash: str
    input_characters: int
    input_tokens: int
    windows: tuple[PlannedWindow, ...]
    model: str = MODEL
    policy: str = POLICY

    def manifest(self):
        return {"model": self.model, "policy": self.policy, "source_hash": self.source_hash,
                "input_characters": self.input_characters, "input_tokens": self.input_tokens,
                "tokenizer_sha256": FILES["tokenizer.json"],
                "windows": [w.manifest() for w in self.windows]}

    @property
    def fingerprint(self):
        return text_hash(canonical(self.manifest()))


class WindowPlanner:
    """Use the pinned BGE tokenizer and prove compatibility before prefix reuse."""
    def __init__(self, directory):
        from tokenizers import Tokenizer
        path = Path(directory) / "mlx-bge/tokenizer.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != FILES["tokenizer.json"]:
            raise ValueError("Window tokenizer does not match the pinned embedding profile")
        self.tokenizer = Tokenizer.from_str(raw.decode())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.prefix_tokenizer = Tokenizer.from_str(self.tokenizer.to_str())
        self.prefix_tokenizer.enable_truncation(max_length=512)

    def plan(self, text):
        full = self.tokenizer.encode(text)
        prefix = self.prefix_tokenizer.encode(text)
        baseline_tokens = (prefix.ids, prefix.type_ids, prefix.attention_mask)
        planned = []
        for window in plan_windows(text, self.tokenizer):
            encoded = self.tokenizer.encode(window.text)
            tokens = (encoded.ids, encoded.type_ids, encoded.attention_mask)
            planned.append(PlannedWindow(window.ordinal, window.start_character,
                window.end_character, window.input_tokens, window.text, text_hash(window.text),
                text_hash(canonical(tokens)), window.ordinal == 1 and tokens == baseline_tokens))
        return WindowPlan(text_hash(text), len(text), len(full.ids), tuple(planned))
