"""Resumable, checksum-verified public model downloads, capped at four connections."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import urllib.request
from .storage import require_space


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ranged_download(url, path, size, sha256, workers=4, block_size=32 * 1024 * 1024, progress=None):
    """No authentication, no automatic retry after throttling, exact Content-Range checks."""
    if not url.startswith("https://huggingface.co/") or not 1 <= workers <= 4:
        raise ValueError("Only public Hugging Face HTTPS downloads with at most four connections are supported")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size and digest(path) == sha256:
        return
    partial, checkpoint = Path(str(path) + ".part"), Path(str(path) + ".progress.json")
    identity = {"url": url, "size": size, "sha256": sha256, "block_size": block_size}
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    if any(state.get(k) != v for k, v in identity.items()) or not partial.exists():
        state = {**identity, "completed": []}
        with partial.open("wb") as f:
            f.truncate(size)
    completed = set(state["completed"])
    abort = threading.Event()
    fd = os.open(partial, os.O_RDWR)
    def fetch(start):
        if abort.is_set():
            return None
        end = min(size, start + block_size) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise ValueError("Server did not honor the requested byte range")
                position = start
                while block := response.read(min(1024 * 1024, end + 1 - position)):
                    require_space(path.parent)
                    view = memoryview(block)
                    while view:
                        written = os.pwrite(fd, view, position)
                        if not written:
                            raise OSError("Unable to write model download")
                        position += written
                        view = view[written:]
                if position != end + 1:
                    raise ValueError("Incomplete model download range")
            return start
        except Exception:
            abort.set()
            raise
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, start) for start in range(0, size, block_size) if start not in completed]
            for future in as_completed(futures):
                start = future.result()
                if start is None:
                    continue
                os.fsync(fd)
                completed.add(start)
                state["completed"] = sorted(completed)
                temporary = Path(str(checkpoint) + ".tmp")
                temporary.write_text(json.dumps(state))
                os.replace(temporary, checkpoint)
                if progress:
                    progress(sum(min(block_size, size - s) for s in completed), size)
    finally:
        os.close(fd)
    if digest(partial) != sha256:
        checkpoint.unlink(missing_ok=True)
        raise ValueError("Model checksum mismatch; the model has not been installed")
    os.replace(partial, path)
    checkpoint.unlink(missing_ok=True)
