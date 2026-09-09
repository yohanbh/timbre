"""Deterministic pre-pass: fix the track manifest and row assignment up front.

Row assignment is a pure function of sorted track ID, so it does not depend on
completion order, thread scheduling, or filesystem readdir order. That is what
lets workers finish in any sequence without changing output bytes.
"""
import sqlite3
from pathlib import Path

import numpy as np

from .embed import DIM, WINDOWS_PER_TRACK, versions

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
    track_id   INTEGER PRIMARY KEY,
    path       TEXT    NOT NULL,
    row_start  INTEGER NOT NULL UNIQUE,
    n_windows  INTEGER,
    status     TEXT    NOT NULL DEFAULT 'pending',
    title      TEXT,
    artist     TEXT,
    genre      TEXT,
    duration   REAL,
    error      TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
"""


def connect(db_path):
    conn = sqlite3.connect(db_path, timeout=60)
    conn.executescript(SCHEMA)
    return conn


def scan(audio_root):
    """All mp3 paths, sorted by integer track id. Opens no audio."""
    paths = Path(audio_root).rglob("*.mp3")
    return sorted(paths, key=lambda p: int(p.stem))


def build(audio_root, db_path, store_path):
    """Create the manifest, allocate the mmap store. Idempotent."""
    conn = connect(db_path)
    existing = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    paths = scan(audio_root)

    if existing == 0:
        rows = [
            (int(p.stem), str(p), i * WINDOWS_PER_TRACK)
            for i, p in enumerate(paths)
        ]
        conn.executemany(
            "INSERT INTO tracks (track_id, path, row_start) VALUES (?, ?, ?)", rows
        )
        meta = {
            "dim": DIM,
            "n_tracks": len(rows),
            "windows_per_track": WINDOWS_PER_TRACK,
            "n_rows": len(rows) * WINDOWS_PER_TRACK,
            **versions(),
        }
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [(k, str(v)) for k, v in meta.items()],
        )
        conn.commit()
    else:
        check_versions(conn)

    n_tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    n_rows = n_tracks * WINDOWS_PER_TRACK

    store = Path(store_path)
    if not store.exists():
        # Allocate full size up front: the .npy header must know the shape, and
        # failed/short tracks keep their zero-filled slot.
        arr = np.lib.format.open_memmap(
            store, mode="w+", dtype=np.float32, shape=(n_rows, DIM)
        )
        arr.flush()
        del arr
    return conn, n_tracks


def check_versions(conn):
    """Refuse to resume a run whose library versions differ -- bit-identity is
    only promised within a fixed software configuration."""
    stored = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    current = versions()
    drift = {k: (stored.get(k), v) for k, v in current.items() if stored.get(k) != v}
    if drift:
        raise RuntimeError(
            f"environment changed since this store was created: {drift}; "
            "byte-identity is not guaranteed across versions"
        )
