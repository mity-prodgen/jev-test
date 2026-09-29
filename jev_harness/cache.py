"""Content-addressed disk cache for judge calls.

Both JevClient and LLMJudgeClient route through get_or_call() so reruns and
re-analysis never re-hit the API. Consistency testing (post 2, step 5) needs
genuinely independent repeated calls, not deduped ones, so it folds a
repeat_index into the key it passes in - the cache itself has no special case
for that, it just hashes whatever key material it's given.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable


def _stable_json(value: Any) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, default=str)


def cache_key(*parts: Any) -> str:
    """Hash arbitrary JSON-able parts into a stable cache key."""
    digest = hashlib.sha256(_stable_json(list(parts)).encode("utf-8")).hexdigest()
    return digest[:32]


class DiskCache:
    def __init__(self, root: str | Path, namespace: str) -> None:
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._path(key).write_text(json.dumps(value, indent=2))

    def get_or_call(self, key: str, call: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        """Returns (result, was_cached).

        Only successful results (result.get("ok") is not False) are persisted -
        a cached error would otherwise survive forever and mask a transient
        failure (rate limit, billing, network blip) as permanent, defeating
        the whole point of being able to just rerun the script.
        """
        cached = self.get(key)
        if cached is not None:
            return cached, True
        result = call()
        if result.get("ok") is not False:
            result.setdefault("cached_at", time.time())
            self.set(key, result)
        return result, False
