"""Fetch only the pinned, checksum-verified tokenizer for integration tests.

No model weights, NRC records, API key, or production data directory is used.
"""
import hashlib
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adams_rag.mlx_embeddings import FILES, REVISION


def main():
    target = Path(__file__).resolve().parents[1] / "tests/.cache/index/mlx-bge/tokenizer.json"
    expected = FILES["tokenizer.json"]
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        print("Test tokenizer already verified")
        return
    url = f"https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/{REVISION}/tokenizer.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("Test tokenizer checksum mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    print(f"Test tokenizer verified ({len(raw):,} bytes; excluded from Git)")


if __name__ == "__main__":
    main()
