"""A disk cache for LLM calls, so repeating a run costs nothing.

One SQLite file maps a hash of the full request (model, settings, prompts) to the
response. Two requests are the same call only if every one of those matches.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ragbasics.generation.llm import Generation, Generator


class DiskCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the worker threads, guarded by a lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)")
            self._db.commit()

    @staticmethod
    def key(request: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    def get(self, request: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM calls WHERE key = ?", (self.key(request),)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, request: dict[str, Any], value: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO calls VALUES (?, ?)",
                (self.key(request), json.dumps(value)),
            )
            self._db.commit()


class CachedGenerator:
    """Wraps a generator so each distinct request is paid for once.

    `sample` is part of the key. The generator has no temperature setting, so asking
    again can give a different answer; sample 1 is a deliberate second draw, used to
    measure that run-to-run noise.
    """

    def __init__(self, inner: Generator, cache: DiskCache, sample: int = 0):
        self.inner = inner
        self.cache = cache
        self.sample = sample
        self.model = inner.model

    def request(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "params": getattr(self.inner, "cache_params", {}),
            "system": system,
            "user": user,
            "sample": self.sample,
        }

    def is_cached(self, system: str, user: str) -> bool:
        return self.cache.get(self.request(system, user)) is not None

    def generate(self, system: str, user: str) -> Generation:
        request = self.request(system, user)
        hit = self.cache.get(request)
        if hit is not None:
            return Generation(**hit, cached=True)
        generation = self.inner.generate(system, user)
        self.cache.put(
            request,
            {
                "text": generation.text,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "stop_reason": generation.stop_reason,
            },
        )
        return generation
