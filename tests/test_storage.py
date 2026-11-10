import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adams_rag.storage import StoragePause, require_space, reserve_gb


class StoragePolicyTests(unittest.TestCase):
    def test_saved_policy_environment_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / "storage.json"
            with patch("adams_rag.storage.POLICY_PATH", path):
                self.assertEqual(reserve_gb(), 2)
                path.write_text(json.dumps({"min_free_gb": .5}))
                self.assertEqual(reserve_gb(), .5)
                with patch.dict(os.environ, {"ADAMS_MIN_FREE_GB": "1.25"}):
                    self.assertEqual(reserve_gb(), 1.25)
                    self.assertEqual(reserve_gb(.75), .75)

    def test_invalid_policy_cannot_disable_reserve(self):
        for value in [.49, -1, float("nan"), float("inf"), True, "unlimited"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                reserve_gb(value)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / "storage.json"
            with patch("adams_rag.storage.POLICY_PATH", path):
                for content in ["[]", "broken", '{"unknown":0.5}', '{"min_free_gb":null}']:
                    path.write_text(content)
                    with self.subTest(content=content), self.assertRaises(ValueError):
                        reserve_gb()

    def test_half_gigabyte_floor_retains_next_write_allowance(self):
        required = 500_000_000 + 32 * 1024 * 1024
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=required)):
            self.assertEqual(require_space(".", .5), required)
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=required - 1)):
            with self.assertRaises(StoragePause):
                require_space(".", .5)
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=700_000_000)):
            self.assertEqual(require_space(".", .5, extra_bytes=200_000_000), 700_000_000)
            with self.assertRaises(StoragePause):
                require_space(".", .5, extra_bytes=200_000_001)
        for extra in [-1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                require_space(".", .5, extra_bytes=extra)
