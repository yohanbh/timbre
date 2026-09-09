"""Acceptance checks for the Phase 0 gate: byte-identical restart."""
import hashlib
import sqlite3
import sys

import numpy as np

from .embed import DIM, WINDOWS_PER_TRACK

CHUNK = 8 << 20  # 8 MiB -- never pull 1 GiB into RAM


def hash_store(store_path):
    """SHA-256 of the whole .npy byte stream, header included.

    The header is constant given a fixed shape, so hashing it also catches an
    accidental reshape. Zero-filled holes are part of the deliverable and must
    be stable too, so hash the entire file rather than only completed rows.
    """
    h = hashlib.sha256()
    with open(store_path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def hash_db(db_path):
    """SHA-256 of the canonical bookkeeping dump. The vector hash alone would
    not catch bookkeeping drift."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT track_id, row_start, n_windows, status FROM tracks ORDER BY track_id"
    ).fetchall()
    conn.close()
    h = hashlib.sha256()
    for r in rows:
        h.update(repr(r).encode())
    return h.hexdigest()


def check_holes_and_norms(db_path, store_path):
    """Every row past n_windows must be exactly zero; every done row must be
    L2-normalized. Catches partial writes a hash flags but cannot explain."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT track_id, row_start, n_windows, status FROM tracks "
        "WHERE status != 'pending' ORDER BY track_id"
    ).fetchall()
    conn.close()
    store = np.load(store_path, mmap_mode="r")

    bad_holes, bad_norms = [], []
    for track_id, row_start, n_win, status in rows:
        n_win = n_win or 0
        hole = store[row_start + n_win:row_start + WINDOWS_PER_TRACK]
        if hole.size and np.any(hole != 0.0):
            bad_holes.append(track_id)
        if n_win:
            norms = np.linalg.norm(store[row_start:row_start + n_win], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                bad_norms.append((track_id, float(norms.min()), float(norms.max())))
    return bad_holes, bad_norms


def report(db_path, store_path):
    conn = sqlite3.connect(db_path)
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM tracks GROUP BY status").fetchall())
    conn.close()
    holes, norms = check_holes_and_norms(db_path, store_path)
    print("status counts :", counts)
    print("store sha256  :", hash_store(store_path))
    print("db sha256     :", hash_db(db_path))
    print("hole integrity:", "OK" if not holes else f"FAIL {holes[:5]}")
    print("norm check    :", "OK" if not norms else f"FAIL {norms[:5]}")
    return not holes and not norms


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "store/timbre.db"
    st = sys.argv[2] if len(sys.argv) > 2 else "store/vectors.npy"
    sys.exit(0 if report(db, st) else 1)
