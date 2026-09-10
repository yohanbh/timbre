"""Print and optionally play the actual ten-second query and matching passages.

Usage:
    PYTHONPATH=src python3 tests/listen.py --track 2 --offset 10 --play
    PYTHONPATH=src python3 tests/listen.py --text "sparse melancholy piano"

Scores are the best segment cosine for each distinct candidate track.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("USE_TF", "0")

import argparse
import shlex
import sqlite3
import subprocess

import numpy as np

from timbre.embed import HOP_SECONDS, Embedder
from timbre.groundtruth import load_layout
from timbre.manifest import check_versions
from timbre.search import exact_track_topk


def play_command(path, offset):
    return ["ffplay", "-v", "error", "-autoexit", "-ss", str(offset),
            "-t", "10", "-nodisp", path]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    query = ap.add_mutually_exclusive_group()
    query.add_argument("--track", type=int, help="query by track id")
    query.add_argument("--text", help="query by text phrase")
    ap.add_argument("--offset", type=int, default=0, help="query window start in seconds")
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--play", action="store_true", help="play each matching passage")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    if a.k < 1 or a.offset < 0 or a.offset % HOP_SECONDS:
        ap.error("k must be positive and offset must be a nonnegative window start")
    if a.text and a.offset:
        ap.error("--offset applies only to audio queries")

    # Read candidate IDs first so metadata includes all of them even while an
    # ingest is adding completed tracks. Existing completed rows are immutable.
    row_ids, owner = load_layout(a.db)
    with sqlite3.connect(f"file:{a.db}?mode=ro", uri=True) as conn:
        if a.text:
            # Audio queries reuse stored vectors; text must use the same model.
            try:
                check_versions(conn)
            except RuntimeError as error:
                ap.error(str(error))
        rows = conn.execute(
            "SELECT track_id, row_start, n_windows, title, artist, genre, path "
            "FROM tracks WHERE status='done' AND n_windows > 0 ORDER BY track_id"
        ).fetchall()
    if not rows:
        ap.error("no embedded tracks in this store")
    tracks = {r[0]: r for r in rows}
    store = np.load(a.store, mmap_mode="r")

    def show_audio(t, offset):
        command = play_command(t[6], offset)
        print(f"    {shlex.join(command)}", flush=True)
        if a.play:
            subprocess.run(command, check=True)

    query_track = None
    if a.text:
        embedder = Embedder()
        embedder.check_text_separation()
        q = embedder.embed_text([a.text])[0]
        print(f'query: "{a.text}"', flush=True)
    else:
        query_track = a.track if a.track is not None else int(
            np.random.default_rng().choice(list(tracks))
        )
        if query_track not in tracks:
            ap.error(f"track {query_track} is not embedded in this store")
        t = tracks[query_track]
        window = a.offset // HOP_SECONDS
        if window >= t[2]:
            ap.error(f"track {query_track}: last window starts at {(t[2] - 1) * HOP_SECONDS}s")
        q = np.asarray(store[t[1] + window])
        print(f"query: {t[3]!r} by {t[4]} [{t[5]}] "
              f"(track {t[0]}, {a.offset}–{a.offset + 10}s)", flush=True)
        show_audio(t, a.offset)

    ids, hit_rows, scores = exact_track_topk(
        store, q, row_ids, owner, k=a.k, exclude_track=query_track
    )
    for rank, (tid, row, score) in enumerate(zip(ids, hit_rows, scores), 1):
        t = tracks[tid]
        offset = int(row - t[1]) * HOP_SECONDS
        print(f"{rank}. {score:.3f}  {t[3]!r} by {t[4]} [{t[5]}] "
              f"(track {tid}, {offset}–{offset + 10}s)", flush=True)
        show_audio(t, offset)


if __name__ == "__main__":
    main()
