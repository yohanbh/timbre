"""Compare exact and HNSW retrieval for one random held-out audio passage.

    PYTHONPATH=src python3 tests/compare_search.py --play
    PYTHONPATH=src python3 tests/compare_search.py --seed 42 --ef-search 128

Both methods use the saved Phase 2 candidate set and exclude the query track
when displaying distinct tracks. Segment recall includes same-track siblings,
matching the Phase 2 benchmark. Timings exclude loading and include aggregation.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("USE_TF", "0")

import argparse
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import time

import numpy as np

from timbre.groundtruth import load_groundtruth, load_layout, owner_lookup, recall_at_k
from timbre.hnsw import HNSW
from timbre.search import exact_track_topk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, help="repeat the same randomly selected query")
    ap.add_argument("--ef-search", type=int, default=64,
                    help="HNSW beam and number of segment hits before track filtering")
    ap.add_argument("-k", type=int, default=5, help="distinct tracks to show per method")
    ap.add_argument("--play", action="store_true", help="play the query and both result lists")
    a = ap.parse_args()
    if a.k < 1 or a.ef_search < max(10, a.k):
        ap.error("require k >= 1 and ef-search >= max(10, k)")

    db, store_path = "store/timbre.db", "store/vectors.npy"
    directory = Path("store/hnsw_phase2")
    phase1 = load_groundtruth("store/groundtruth.npz", db, store_path)
    with np.load(directory / "groundtruth.npz", allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        if any(metadata.get(key) != str(value) for key, value in phase1.items()
               if key.startswith("source_")):
            raise RuntimeError("Phase 2 ground truth does not match the current store")
        candidates, query_rows, truth = (
            cache[key].copy() for key in ("candidate_rows", "query_rows", "gt_ids")
        )
    rows, owners = load_layout(db)
    lut = owner_lookup(rows, owners)
    if (not np.isin(candidates, rows).all() or
            not np.isin(query_rows, phase1["query_rows"]).all() or
            np.intersect1d(candidates, query_rows).size):
        raise RuntimeError("invalid Phase 2 candidate/query membership")
    store = np.load(store_path, mmap_mode="r")
    index = HNSW.load(directory / "M8_efC80_shuffled_seed20260909.npz",
                      np.array(store[candidates]))
    if index.size != len(candidates) or not np.array_equal(index.ids, candidates):
        raise RuntimeError("graph membership does not match the Phase 2 oracle")

    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        tracks = {r[0]: r for r in conn.execute(
            "SELECT track_id, row_start, title, artist, genre, path "
            "FROM tracks WHERE status='done' AND n_windows > 0"
        )}
    query_index = int(np.random.default_rng(a.seed).integers(len(query_rows)))
    query_row = int(query_rows[query_index])
    query_track = int(lut[query_row])
    query = np.asarray(store[query_row])

    def exact():
        result = exact_track_topk(store, query, candidates, lut[candidates],
                                  k=a.k, exclude_track=query_track)
        return list(zip(*result))

    def approximate():
        ids, scores = index.search(query, k=a.ef_search, ef_search=a.ef_search)
        seen, hits = {query_track}, []
        for row, score in zip(ids, scores):
            track = int(lut[row])
            if track not in seen:
                seen.add(track)
                hits.append((track, row, score))
                if len(hits) == a.k:
                    break
        return hits, ids

    # Warm each path once; this is a single-query comparison, not a benchmark.
    exact()
    approximate()
    started = time.perf_counter()
    exact_hits = exact()
    exact_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    ann_hits, segment_ids = approximate()
    ann_ms = (time.perf_counter() - started) * 1000

    def show(track, row, prefix):
        _, row_start, title, artist, genre, path = tracks[int(track)]
        offset = int(row) - row_start
        print(f"{prefix}{title!r} by {artist} [{genre}] "
              f"(track {track}, {offset}\u2013{offset + 10}s)", flush=True)
        command = ["ffplay", "-v", "error", "-autoexit", "-ss", str(offset),
                   "-t", "10", "-nodisp", path]
        print(f"    {shlex.join(command)}", flush=True)
        if a.play:
            subprocess.run(command, check=True)

    print(f"Both methods: {len(candidates):,} candidate segments; "
          f"HNSW M={index.M}, efConstruction={index.ef_construction}, "
          f"efSearch={a.ef_search}", flush=True)
    show(query_track, query_row, "Query: ")
    for label, hits, elapsed in (("Phase 1: exact", exact_hits, exact_ms),
                                 ("Phase 2: HNSW", ann_hits, ann_ms)):
        print(f"\n{label} \u2014 {elapsed:.3f} ms, {len(hits)} distinct tracks", flush=True)
        for rank, (track, row, score) in enumerate(hits, 1):
            show(track, row, f"{rank}. {score:.4f}  ")

    overlap = len({int(h[0]) for h in exact_hits} & {int(h[0]) for h in ann_hits})
    recall = recall_at_k(segment_ids[None, :], truth[query_index:query_index + 1], 10)
    print(f"\nTrack overlap: {overlap}/{len(exact_hits)} exact results recovered")
    print(f"Segment recall@10: {recall:.0%} (includes same-track siblings)")
    print("Timings are one warm query including track aggregation; exclude all loading.")
    if len(ann_hits) < len(exact_hits):
        print("HNSW returned fewer distinct tracks; rerun with a larger --ef-search.")


if __name__ == "__main__":
    main()
