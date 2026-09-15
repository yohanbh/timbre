"""Reuse embeddings only when source and target MP3 bytes/configuration match.

The target manifest must already contain the complete corpus. Row positions are
remapped by track ID, and vector writes become durable before done markers commit.
"""
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3

import numpy as np

from . import manifest
from .embed import DIM, WINDOWS_PER_TRACK
from .verify import check_holes_and_norms


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1 << 20):
            h.update(chunk)
    return h.digest()


def seed(source_db, source_store, target_db, target_store):
    with closing(sqlite3.connect(Path(source_db).resolve().as_uri() + "?mode=ro", uri=True)) as source, \
            closing(manifest.connect(target_db)) as target:
        manifest.check_versions(source)
        manifest.check_versions(target)
        if source.execute("SELECT COUNT(*) FROM tracks WHERE status='pending'").fetchone()[0]:
            raise ValueError("source ingest is incomplete")
        holes, norms = check_holes_and_norms(source_db, source_store)
        if holes or norms:
            raise ValueError("source vector integrity check failed")
        source_vectors = np.load(source_store, mmap_mode="r")
        target_vectors = np.load(target_store, mmap_mode="r+")
        if source_vectors.shape[1] != DIM or target_vectors.shape[1] != DIM:
            raise ValueError("unexpected vector dimension")
        pending = {tid: (path, row) for tid, path, row in target.execute(
            "SELECT track_id,path,row_start FROM tracks WHERE status='pending'"
        )}
        seeded = int(dict(target.execute("SELECT key,value FROM meta")).get("seeded_tracks", 0))
        copied, mismatched, updates = 0, 0, []
        fd = os.open(target_store, os.O_RDONLY)
        try:
            def commit_batch():
                nonlocal seeded
                if not updates:
                    return
                target_vectors.flush()
                os.fsync(fd)
                target.executemany("UPDATE tracks SET status='done',n_windows=?,error=NULL WHERE track_id=?", updates)
                seeded += len(updates)
                target.execute("INSERT OR REPLACE INTO meta VALUES ('seeded_tracks',?)", (str(seeded),))
                target.commit()
                updates.clear()

            for tid, source_path, row, n_windows in source.execute(
                "SELECT track_id,path,row_start,n_windows FROM tracks WHERE status='done' ORDER BY track_id"
            ):
                if tid not in pending:
                    continue
                target_path, target_row = pending[tid]
                if not Path(source_path).is_file() or file_hash(source_path) != file_hash(target_path):
                    mismatched += 1
                    continue
                target_vectors[target_row:target_row + WINDOWS_PER_TRACK] = source_vectors[row:row + WINDOWS_PER_TRACK]
                updates.append((n_windows, tid))
                copied += 1
                if len(updates) == 128:
                    commit_batch()
                if copied % 1000 == 0:
                    print(f"  reused {copied} matching tracks", flush=True)
            commit_batch()
        finally:
            os.close(fd)
        return {"copied_this_run": copied, "seeded_tracks_total": seeded, "audio_mismatches": mismatched}
