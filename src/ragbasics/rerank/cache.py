"""A disk cache for reranker scores.

One SQLite file maps a key to one number. The key is a hash of everything that decides
the score: the model and its revision, the input limit, the prompt template if the model
has one, the question and the passage text (built in `CrossEncoderReranker`). A score is
a function of the pair alone, so one pass over 100 candidates also serves depths 20 and
50, and a repeat run is free and gives the same numbers to the last bit.
"""

import hashlib
import sqlite3
import threading
from pathlib import Path

SQL_VARIABLES = 500  # keys per SELECT, under SQLite's limit on bound values


def pair_key(identity: str, question: str, passage: str) -> str:
    parts = (identity, question, passage)
    return hashlib.sha256(b"\x00".join(part.encode() for part in parts)).hexdigest()


class RerankCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the evaluator's worker threads, guarded by a lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL)")
            self._db.commit()

    def get_many(self, keys: list[str]) -> dict[str, float]:
        found: dict[str, float] = {}
        with self._lock:
            for i in range(0, len(keys), SQL_VARIABLES):
                batch = keys[i : i + SQL_VARIABLES]
                marks = ",".join("?" * len(batch))
                rows = self._db.execute(
                    f"SELECT key, score FROM scores WHERE key IN ({marks})", batch
                )
                found.update(rows)
        return found

    def put_many(self, scores: dict[str, float]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO scores VALUES (?, ?)",
                [(key, float(score)) for key, score in scores.items()],
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
