"""Build and cache the Phase 1 ground truth.

    PYTHONPATH=src python3 -m timbre.build_groundtruth

Writes store/groundtruth.npz. Rerunning with the same seed and store must
produce identical bytes; that is the Phase 1 analogue of the Phase 0 gate.
"""
import sys
import time

import numpy as np

from .groundtruth import (N_QUERIES, SEED, TOP_K, exact_topk, load_layout,
                          owner_lookup, sample_queries)


def main(db="store/timbre.db", store_path="store/vectors.npy",
         out="store/groundtruth.npz"):
    row_ids, owner = load_layout(db)
    lut = owner_lookup(row_ids, owner)
    store = np.load(store_path, mmap_mode="r")
    print(f"corpus: {row_ids.size} populated rows, {np.unique(owner).size} tracks")

    q_rows, q_tracks, q_genres = sample_queries(db, N_QUERIES, SEED)
    print(f"queries: {q_rows.size} genre-stratified segments (seed {SEED})")

    results = {}
    for name, excl in (("", False), ("_nosib", True)):
        t0 = time.time()
        ids, sims = exact_topk(store, q_rows, row_ids, lut, q_tracks,
                               k=TOP_K, exclude_siblings=excl)
        results[f"gt_ids{name}"] = ids
        results[f"gt_sims{name}"] = sims
        label = "sibling-excluded" if excl else "unfiltered"
        print(f"  {label:16s} top-{TOP_K} in {time.time() - t0:.1f}s  "
              f"mean sim@1={sims[:, 0].mean():.4f} @10={sims[:, 9].mean():.4f}")

    np.savez(out, query_rows=q_rows, query_tracks=q_tracks,
             query_genres=np.asarray(q_genres, dtype=str), seed=SEED, **results)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
