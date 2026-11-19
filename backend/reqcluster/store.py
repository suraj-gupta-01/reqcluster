"""SQLite-backed embedding cache. Never fails a job: errors degrade to misses."""

import hashlib
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
_CHUNK = 900


def key(model: str, revision: str, text: str) -> str:
    return hashlib.sha256("\x1f".join([model, revision, text]).encode("utf-8")).hexdigest()


class EmbeddingStore:
    def __init__(self, path: Path, dim: int):
        self.path = Path(path)
        self.dim = dim
        self._read_warned = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as con, con:
                con.execute("CREATE TABLE IF NOT EXISTS vec(key TEXT PRIMARY KEY, data BLOB NOT NULL)")
        except (sqlite3.Error, OSError) as exc:
            log.warning("embedding cache init failed: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=5)
        con.execute("PRAGMA busy_timeout=5000")
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        try:
            with closing(self._connect()) as con:
                for i in range(0, len(keys), _CHUNK):
                    chunk = keys[i : i + _CHUNK]
                    marks = ",".join("?" * len(chunk))
                    rows = con.execute(f"SELECT key, data FROM vec WHERE key IN ({marks})", chunk)
                    for k, blob in rows:
                        v = np.frombuffer(blob, dtype=np.float32).copy()
                        if v.shape[0] == self.dim:
                            out[k] = v
        except sqlite3.Error as exc:
            if not self._read_warned:
                self._read_warned = True
                log.warning("embedding cache read failed: %s", exc)
            return {}
        return out

    def put_many(self, keys: list[str], vecs: np.ndarray) -> None:
        rows = [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in zip(keys, vecs, strict=True)]
        try:
            with closing(self._connect()) as con, con:
                con.executemany("INSERT OR IGNORE INTO vec(key, data) VALUES (?, ?)", rows)
        except sqlite3.Error as exc:
            log.warning("embedding cache write failed: %s", exc)
