"""Leave room for the operating system and the next atomic write."""
import json
import math
import os
from pathlib import Path
import shutil

POLICY_PATH = Path(__file__).resolve().parent.parent / ".storage.json"


def reserve_gb(value=None):
    """Explicit override, environment, saved local policy, then the 2 GB default."""
    if value is None:
        value = os.environ.get("ADAMS_MIN_FREE_GB")
        if value is None:
            policy = json.loads(POLICY_PATH.read_text()) if POLICY_PATH.exists() else {}
            if not isinstance(policy, dict) or set(policy) - {"min_free_gb"}:
                raise ValueError("Invalid local storage policy")
            value = policy.get("min_free_gb", 2)
    try:
        if isinstance(value, bool):
            raise ValueError
        value = float(value)
        if not math.isfinite(value) or value < .5:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("Use a finite storage reserve of at least 0.5 GB") from None
    return value


class StoragePause(RuntimeError):
    """A resumable stop, never an invitation to delete source data."""


def require_space(path, min_free_gb=None, extra_bytes=32 * 1024 * 1024):
    min_free_gb = reserve_gb(min_free_gb)
    if not isinstance(extra_bytes, (int, float)) or not math.isfinite(extra_bytes) or extra_bytes < 0:
        raise ValueError("Use nonnegative finite write headroom")
    free = shutil.disk_usage(path).free
    required = int(min_free_gb * 1_000_000_000) + extra_bytes
    if free < required:
        raise StoragePause(f"Storage pause: {free / 1e9:.2f} GB free; preserve {min_free_gb:g} GB plus space for the next write")
    return free
