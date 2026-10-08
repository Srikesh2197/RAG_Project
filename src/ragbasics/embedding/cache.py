"""A disk cache for embedding vectors.

One SQLite file maps a key to a vector stored as raw float32 bytes. The key is a hash
of the model, anything put in front of the text, and the text (built in
`CachingEmbedder`). Vectors are stored at the model's full size and before any
normalisation, so one cached vector serves every truncated length.
"""

import sqlite3
import threading
from pathlib import Path

import numpy as np

SQL_VARIABLES = 500  # keys per SELECT, under SQLite's limit on bound values


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the evaluator's worker threads, guarded by a lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)"
            )
            self._db.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), SQL_VARIABLES):
                batch = keys[i : i + SQL_VARIABLES]
                marks = ",".join("?" * len(batch))
                rows = self._db.execute(
                    f"SELECT key, vector FROM vectors WHERE key IN ({marks})", batch
                )
                for key, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, vectors: dict[str, np.ndarray]) -> None:
        rows = [(key, np.asarray(v, dtype=np.float32).tobytes()) for key, v in vectors.items()]
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", rows)
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
