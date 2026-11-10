import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adams_rag.config import load_api_key


class CredentialTests(unittest.TestCase):
    def test_config_loads_owner_only_secret_without_embedding_it(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            secret = root / "key"
            secret.write_text("test-only-value\n")
            secret.chmod(0o600)
            config = root / "config.json"
            config.write_text(json.dumps({"api_key_file": str(secret)}))
            self.assertEqual(load_api_key(config), "test-only-value")
            self.assertNotIn("test-only-value", config.read_text())
            with patch.dict(os.environ, {"ADAMS_API_KEY": "environment-value"}):
                self.assertEqual(load_api_key(config), "environment-value")

    def test_insecure_secret_permissions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            secret = Path(tmp) / "key"
            secret.write_text("test-only-value")
            secret.chmod(0o644)
            with patch.dict(os.environ, {"ADAMS_API_KEY_FILE": str(secret)}):
                with self.assertRaisesRegex(ValueError, "owner-only"):
                    load_api_key()

    def test_no_config_is_not_a_credential(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(load_api_key(Path(tmp) / "absent.json"))
