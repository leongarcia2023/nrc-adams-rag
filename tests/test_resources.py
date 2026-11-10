import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adams_rag.download import ranged_download
from adams_rag.models import model_ready


class ResourceTests(unittest.TestCase):
    def test_partial_model_is_not_reported_as_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            self.assertFalse(model_ready(root))
            (root / "resource-manifest.json").write_text(json.dumps({"files": [{"path": "model.safetensors", "bytes": 4}]}))
            (root / "model.safetensors").write_bytes(b"part")
            self.assertTrue(model_ready(root))
            (root / "model.safetensors").write_bytes(b"par")
            self.assertFalse(model_ready(root))

    def test_download_checks_ranges_hash_and_reuses_verified_file(self):
        content = b"The locally installed model must be complete."
        checksum = hashlib.sha256(content).hexdigest()
        requests = []
        def open_range(request, **kwargs):
            requests.append(request)
            start, end = map(int, request.headers["Range"][6:].split("-"))
            response = io.BytesIO(content[start:end + 1])
            response.status = 206
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
            return response
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", side_effect=open_range):
            target = Path(directory) / "weights"
            ranged_download("https://huggingface.co/test/model", target, len(content), checksum, block_size=10)
            self.assertEqual(target.read_bytes(), content)
            count = len(requests)
            ranged_download("https://huggingface.co/test/model", target, len(content), checksum, block_size=10)
            self.assertEqual(len(requests), count)
            with self.assertRaisesRegex(ValueError, "checksum"):
                ranged_download("https://huggingface.co/test/model", Path(directory) / "bad", len(content), "wrong", block_size=10)
            self.assertFalse((Path(directory) / "bad").exists())

    def test_server_ignoring_range_cannot_corrupt_installed_model(self):
        response = io.BytesIO(b"some whole file")
        response.status, response.headers = 200, {}
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", return_value=response):
            path = Path(directory) / "weights"
            with self.assertRaisesRegex(ValueError, "byte range"):
                ranged_download("https://huggingface.co/test/model", path, 15, "hash", workers=1)
            self.assertFalse(path.exists())
