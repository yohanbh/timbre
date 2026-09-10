"""Build and cache the Phase 1 ground truth.

    PYTHONPATH=src python3 -m timbre.build_groundtruth

Writes store/groundtruth.npz. Rerunning against the same store must reproduce
it byte for byte; that is the Phase 1 analogue of the Phase 0 gate.

Thread count is pinned here, before numpy is imported, for the same reason
__main__.py pins it before torch: BLAS sizes its pool at import time, so setting
it later has no effect. It matters more than it looks. On the original checkpoint,
the cached top-100 is bit-identical at BLAS_THREADS=4 but differs at 1 or 2
threads -- and not only in the similarity values: *neighbour ids* change too,
because a different reduction order moves the last bits and this corpus packs
neighbours at 0.979 similarity, so near-ties reorder. Ground truth that silently
depends on an ambient env var is not ground truth, hence the pin and the
recorded value.
"""
import os

BLAS_THREADS = "4"
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = BLAS_THREADS

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from .groundtruth import (N_QUERIES, SEED, TOP_K, exact_topk,  # noqa: E402
                          load_layout, owner_lookup, sample_queries, source_identity)


def main(db="store/timbre.db", store_path="store/vectors.npy",
         out="store/groundtruth.npz"):
    identity = source_identity(db, store_path)
    row_ids, owner = load_layout(db)
    lut = owner_lookup(row_ids, owner)
    store = np.load(store_path, mmap_mode="r")
    print(f"corpus: {row_ids.size} populated rows, {np.unique(owner).size} tracks")

    q_rows, q_tracks, q_genres = sample_queries(db, N_QUERIES, SEED)
    print(f"queries: {q_rows.size} genre-stratified segments (seed {SEED}, "
          f"{BLAS_THREADS} BLAS threads)")

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
             query_genres=np.asarray(q_genres, dtype=str), seed=SEED,
             blas_threads=int(BLAS_THREADS), **identity, **results)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
