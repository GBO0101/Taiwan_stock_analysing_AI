"""Local TTL file cache for Step 4 enumeration data sources.

TWSE ISIN listings and TIP index constituents change infrequently. We cache
them to a local JSON file with a TTL (default 7 days) to avoid hammering the
public endpoints on every pipeline run. Supply-chain LLM results are NOT
cached (query-dependent, fast-changing).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent / "cache"


class CacheError(Exception):
    """Cache read/write errors."""


def _cache_path(key: str) -> Path:
    """Return the cache file path for a key (sanitized)."""
    safe = "".join(c for c in key if c.isalnum() or c in "._-") or "default"
    return CACHE_DIR / f"{safe}.json"


def get_cache(key: str, ttl_seconds: int = 7 * 24 * 3600) -> Any | None:
    """Return cached value for ``key`` if present and fresh, else None."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise CacheError(f"Failed to read cache {path}: {e}") from e

    stored_at = payload.get("_stored_at", 0)
    if time.time() - stored_at > ttl_seconds:
        return None
    return payload.get("data")


def set_cache(key: str, value: Any) -> None:
    """Store ``value`` under ``key`` with a timestamp for TTL expiry."""
    path = _cache_path(key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"_stored_at": time.time(), "data": value}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise CacheError(f"Failed to write cache {path}: {e}") from e
