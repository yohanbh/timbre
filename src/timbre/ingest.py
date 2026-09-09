"""Resumable ingest: decode -> mel (workers) -> CLAP (GPU) -> mmap + SQLite.

Crash safety hinges on write ordering:
    write vectors -> flush -> fsync -> THEN commit the done-marker
The reverse is unsafe: commit first, die before fsync, and restart skips a track
whose rows are permanently zeroed -- silent corruption, the worst failure mode.
With this ordering the worst case is redundant re-embedding that rewrites
byte-identical data over the same rows.
"""
import os
import signal
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import manifest
from .audio import DecodeError, decode, windows
from .embed import DIM, WINDOWS_PER_TRACK, Embedder

N_WORKERS = 12          # leave headroom for GPU feeder + ffmpeg subprocesses
QUEUE_DEPTH = 8         # bounds in-flight mel at ~41 MiB
FSYNC_EVERY = 1         # checkpoint interval in tracks

_processor = None


def _worker_init():
    """Workers do decode AND mel: mel is the expensive part (~1394 ms/track vs
    94 ms decode), so leaving it in the parent would recreate the bottleneck."""
    global _processor
    os.environ["OMP_NUM_THREADS"] = "1"
    from transformers import ClapProcessor
    from .embed import MODEL_ID, MODEL_REVISION
    _processor = ClapProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def _prepare(task):
    """Decode + mel for one track. Returns (track_id, features, n_windows, error)."""
    track_id, path = task
    try:
        audio = decode(path)
    except DecodeError as e:
        return track_id, None, 0, f"decode: {e}"
    wins = windows(audio)
    if not wins:
        return track_id, None, 0, "short"
    from .embed import assert_window_len
    for w in wins:
        assert_window_len(w)
    feats = _processor(audios=wins, sampling_rate=48000, return_tensors="pt")
    return track_id, feats, len(wins), None


def run(audio_root, db_path, store_path, limit=None):
    embedder = Embedder()
    embedder.check_dim()  # never size the store on a guessed width

    conn, _ = manifest.build(audio_root, db_path, store_path)
    pending = conn.execute(
        "SELECT track_id, path, row_start FROM tracks WHERE status='pending' ORDER BY track_id"
    ).fetchall()
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print("nothing pending")
        return

    rows_by_id = {t: r for t, _, r in pending}
    store = np.lib.format.open_memmap(store_path, mode="r+")
    store_fd = os.open(store_path, os.O_RDONLY)
    print(f"ingesting {len(pending)} tracks", flush=True)

    done = 0
    tasks = [(t, p) for t, p, _ in pending]
    with ProcessPoolExecutor(N_WORKERS, initializer=_worker_init) as pool:
        for track_id, feats, n_win, error in pool.map(_prepare, tasks, chunksize=1):
            row_start = rows_by_id[track_id]
            if error is not None:
                status = "short" if error == "short" else "failed"
                # Reserved slot stays zero-filled; SQLite records why.
                conn.execute(
                    "UPDATE tracks SET status=?, n_windows=0, error=? WHERE track_id=?",
                    (status, None if status == "short" else error, track_id),
                )
                conn.commit()
                continue

            vecs = embedder.embed_features(feats)
            assert vecs.shape == (n_win, DIM), vecs.shape

            # 1. write real vectors, 2. zero-fill the rest of the reserved block
            store[row_start:row_start + n_win] = vecs
            if n_win < WINDOWS_PER_TRACK:
                store[row_start + n_win:row_start + WINDOWS_PER_TRACK] = 0.0

            done += 1
            if done % FSYNC_EVERY == 0:
                store.flush()          # msync
                os.fsync(store_fd)     # durable
            # only now is it safe to mark done
            conn.execute(
                "UPDATE tracks SET status='done', n_windows=?, error=NULL WHERE track_id=?",
                (n_win, track_id),
            )
            conn.commit()
            if done % 50 == 0:
                print(f"  {done}/{len(pending)}", flush=True)

    store.flush()
    os.fsync(store_fd)
    os.close(store_fd)
    conn.close()
    print(f"done: {done} embedded")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", default="data/fma_medium")
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.audio_root, a.db, a.store, a.limit)
