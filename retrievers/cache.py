from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class SimpleCache:
    """Lightweight JSON-on-disk cache with TTL."""

    def __init__(self, path: str, ttl_hours: float = 24):
        self.path = path
        self.ttl_seconds = ttl_hours * 3600
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._data = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception:
            self._data = {}

    def _persist(self) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        os.replace(tmp_path, self.path)

    def _is_fresh(self, ts: float) -> bool:
        return (time.time() - ts) < self.ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        entry = self._data.get(key)
        if not entry:
            return None
        ts = entry.get("ts", 0)
        if not self._is_fresh(ts):
            self._data.pop(key, None)
            return None
        return entry.get("value")

    def set(self, key: str, value: Any) -> None:
        self._data[key] = {"ts": time.time(), "value": value}
        self._persist()


__all__ = ["SimpleCache"]
