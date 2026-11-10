"""Opt-in, one-page-at-a-time OCR with no remote service or model download."""
import importlib.util
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

from .inference_guard import run_guarded
from .storage import require_space
from .pdf_policy import DEFAULT_PDF_BYTES, file_sha256, validate_pdf_bytes

POLICY_VERSION = 1


class PDFSourceChanged(ValueError):
    def __init__(self, before, after):
        super().__init__("PDF changed during extraction; cached results cannot be reused")
        self.changed_fields = [name for name, old, new in zip(
            ("device", "inode", "bytes", "mtime_ns", "ctime_ns"), before, after) if old != new]


class PDFSourceSnapshot:
    """Hash once per import; reject source changes before reusing page caches."""
    def __init__(self, path, max_bytes=DEFAULT_PDF_BYTES):
        self.max_bytes = validate_pdf_bytes(max_bytes)
        self.path = Path(path).resolve(strict=True)
        self.identity = self._identity()
        self.metadata_rechecks = 0
        if self.identity[2] > self.max_bytes:
            raise ValueError("OCR file allowance exceeded")
        self.sha256 = file_sha256(self.path, self.max_bytes)
        self.check()

    def _identity(self):
        state = self.path.stat()
        return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)

    def check(self):
        current = self._identity()
        if current == self.identity:
            return
        # A permission/xattr change can advance ctime while file contents stay
        # unchanged. Accept only that narrow case, with a full bounded rehash and
        # stable metadata across the read. Size, mtime or file replacement stops.
        if current[:4] == self.identity[:4]:
            checksum = file_sha256(self.path,self.max_bytes)
            if self._identity() == current and checksum == self.sha256:
                self.identity = current
                self.metadata_rechecks += 1
                return
        raise PDFSourceChanged(self.identity,current)


def valid_page(result, number):
    return (isinstance(result, dict) and type(result.get("page")) is int and result.get("page") == number
            and isinstance(result.get("text"), str) and isinstance(result.get("lines"), list)
            and result.get("method") == "apple_vision_ocr")


def cached_page(path, number, storage_path, cancelled=None, before_recognize=None, snapshot=None):
    """Publish completed pages independently, so an interrupted PDF resumes OCR."""
    path = Path(path)
    if cancelled and cancelled():
        raise InterruptedError("PDF import cancelled")
    source = snapshot or PDFSourceSnapshot(path)
    if not isinstance(source,PDFSourceSnapshot) or source.path != path.resolve():
        raise ValueError("Page cache source snapshot does not match the PDF")
    source.check()
    source_hash = source.sha256
    policy = {"version": POLICY_VERSION, "os": platform.mac_ver()[0], "recognition_revision": 3}
    cache_key = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
    target = Path(storage_path) / "ocr_pages" / source_hash[:2] / source_hash / f"{number}-{cache_key}.json.gz"
    if target.exists():
        try:
            with gzip.open(target, "rb") as cached:
                data = cached.read(1024 * 1024 + 1)
            if len(data) <= 1024 * 1024:
                envelope = json.loads(data)
                if not isinstance(envelope, dict):
                    raise ValueError("Invalid page cache")
                result = envelope.get("result")
                canonical = json.dumps(result, sort_keys=True, ensure_ascii=False).encode()
                if (envelope.get("source_hash") == source_hash and envelope.get("policy") == policy
                        and envelope.get("result_hash") == hashlib.sha256(canonical).hexdigest() and valid_page(result, number)):
                    source.check()
                    return result
        except (OSError, EOFError, ValueError, TypeError):
            pass  # A corrupt derived page is rebuilt from the retained PDF.
    source.check()
    if before_recognize:
        before_recognize(number)
    result = recognize_page(path, number, storage_path, cancelled=cancelled, max_bytes=source.max_bytes)
    if not valid_page(result, number):
        raise ValueError("OCR returned an invalid page result")
    source.check()
    if cancelled and cancelled():
        raise InterruptedError("PDF import cancelled")
    canonical = json.dumps(result, sort_keys=True, ensure_ascii=False).encode()
    envelope = {"source_hash": source_hash, "policy": policy,
                "result_hash": hashlib.sha256(canonical).hexdigest(), "result": result}
    encoded = json.dumps(envelope, ensure_ascii=False).encode()
    if len(encoded) > 1024 * 1024:
        raise ValueError("Recognized page exceeds the cache allowance")
    require_space(storage_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(gzip.compress(encoded))
        source.check()
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result


def recognize_page(path, number, storage_path, cancelled=None, max_bytes=DEFAULT_PDF_BYTES):
    validate_pdf_bytes(max_bytes)
    if Path(path).stat().st_size > max_bytes:
        raise ValueError("OCR file allowance exceeded")
    if sys.platform != "darwin" or importlib.util.find_spec("Vision") is None:
        raise RuntimeError("Local OCR requires macOS and the optional adams-rag[ocr] dependency")
    def check():
        for location in {Path(storage_path).resolve(), Path(tempfile.gettempdir()).resolve()}:
            require_space(location, extra_bytes=512 * 1024 * 1024 + 2 * Path(path).stat().st_size)
    check()
    with (Path(storage_path) / "ocr.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another local OCR call is running") from None
        result = run_guarded([sys.executable, "-m", "adams_rag.ocr_worker"],
                             {"path": str(Path(path).resolve(strict=True)), "page": number,
                              "parent_pid": os.getpid(), "max_bytes": max_bytes},
                             preflight=check, monitor=check, timeout=120, cancelled=cancelled)
    if not valid_page(result, number):
        raise ValueError("OCR returned an invalid page result")
    return result
