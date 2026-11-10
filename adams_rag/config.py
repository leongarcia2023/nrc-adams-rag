"""Local credentials stay outside the distributable application folder."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path


def load_api_key(config_path=None):
    key = os.environ.get("ADAMS_API_KEY", "").strip()
    if key:
        return key
    key_path = os.environ.get("ADAMS_API_KEY_FILE")
    if not key_path:
        config_path = Path(config_path) if config_path else Path(__file__).resolve().parent.parent / ".local.json"
        if not config_path.exists():
            return None
        config = json.loads(config_path.read_text())
        key_path = config.get("api_key_file")
    if not key_path:
        return None
    path = Path(key_path).expanduser()
    if not path.is_absolute():
        raise ValueError("The API key file must have an absolute path")
    with path.open() as file:
        permissions = os.fstat(file.fileno())
        if not stat.S_ISREG(permissions.st_mode):
            raise ValueError("The API key must be stored in a regular file")
        if os.name == "posix" and (permissions.st_mode & 0o077 or permissions.st_uid != os.getuid()):
            raise ValueError("The API key file must be owned by you with owner-only permissions")
        key = file.read(4097).strip()
    if not key or len(key) > 4096 or any(c.isspace() for c in key):
        raise ValueError("The API key file is empty or malformed")
    return key
