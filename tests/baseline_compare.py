"""Calibrate our hand-written HNSW against hnswlib and FAISS-HNSW.

    USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 tests/baseline_compare.py \
      --graph store/hnsw_construction_large/cpp_direct \
      --oracle store/large/index_groundtruth.npz --store store/large/vectors.npy

These libraries are reference baselines and never appear in the serving path
(TIMBRE_SPEC.md, Guardrails). The deliverable is not beating hnswlib — it is
years of SIMD tuning — but saying precisely where the time goes.

Fairness rules, because an unfair baseline is worse than none:

  * identical candidate vectors, identical held-out queries, identical oracle
  * the same M and efConstruction, and each efSearch measured on all three
  * one thread everywhere: OMP_NUM_THREADS=1, faiss.omp_set_num_threads(1),
    hnswlib num_threads=1, and one query per call rather than a batch
  * every index built fresh here, so build time is comparable; our saved graph
    is loaded separately to confirm the fresh build reproduces its recall
  * cosine space for all three, over L2-normalized vectors, so inner product
    and cosine rank identically

Recall is scored against the same exact oracle used everywhere else in the
project, so the number means the same thing it does in the other reports.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import argparse
import json
import platform
import resource
import time

import numpy as np

from timbre.groundtruth import recall_at_k
from timbre.native_hnsw import NativeHNSW, NativeHNSWBuilder


def timed_queries(search, queries, warmup=50):
    """Per-query wall time; one query per call, never a batch."""
    for q in queries[:warmup]:
        search(q)
    milliseconds, ids = [], []
    for q in queries:
        started = time.perf_counter()
        found = search(q)
        milliseconds.append((time.perf_counter() - started) * 1000)
        ids.append(found)
    return np.array(ids), np.array(milliseconds)


def summarize(ids, milliseconds, truth, k):
    return {
        "recall_at_10": float(recall_at_k(ids, truth, k)),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p99_ms": float(np.percentile(milliseconds, 99)),
        "queries_per_second": float(1000 / np.mean(milliseconds)),
    }


def build_ours(vectors, ids, M, ef_construction, seed):
    builder = NativeHNSWBuilder(vectors, ids, M=M, ef_construction=ef_construction,
                                seed=seed)
    order = np.random.default_rng(seed + 1).permutation(len(vectors))
    builder.build(order)
    return builder.freeze()


def build_hnswlib(vectors, ids, M, ef_construction, seed):
    import hnswlib

    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), M=M,
                     ef_construction=ef_construction, random_seed=seed)
    index.set_num_threads(1)
    index.add_items(vectors, ids, num_threads=1)
    return index


def build_faiss(vectors, M, ef_construction):
    import faiss

    faiss.omp_set_num_threads(1)
    # METRIC_INNER_PRODUCT over normalized vectors ranks identically to cosine.
    index = faiss.IndexHNSWFlat(vectors.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True, help="our saved direct graph")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--ef-construction", type=int, default=80)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap candidates for a quick run; 0 uses every candidate")
    ap.add_argument("--out", default="docs/baseline_comparison_results.json")
    a = ap.parse_args()

    with np.load(a.oracle, allow_pickle=False) as arrays:
        candidates = np.asarray(arrays["candidate_rows"], dtype=np.int64)
        query_rows = np.asarray(arrays["query_rows"], dtype=np.int64)
        truth = np.asarray(arrays["gt_ids"])
    store = np.load(a.store, mmap_mode="r")
    if a.limit:
        # Subsetting invalidates the oracle, so recompute exact neighbours for
        # the subset rather than scoring against neighbours that are no longer
        # candidates.
        candidates = candidates[:a.limit]
        queries = np.array(store[query_rows])
        block = np.array(store[candidates])
        truth = np.empty((len(queries), max(a.k, 10)), dtype=np.int64)
        for i, q in enumerate(queries):
            scores = block @ (q / np.linalg.norm(q))
            top = np.argpartition(-scores, truth.shape[1])[:truth.shape[1]]
            truth[i] = candidates[top[np.argsort(-scores[top])]]
    else:
        queries = np.array(store[query_rows])
    vectors = np.ascontiguousarray(np.array(store[candidates]), dtype=np.float32)
    del store
    print(f"{len(vectors):,} candidates, {len(queries)} queries, "
          f"M={a.m} efC={a.ef_construction}, one thread", flush=True)

    report = {"candidates": len(vectors), "queries": len(queries), "k": a.k,
              "M": a.m, "ef_construction": a.ef_construction, "seed": a.seed,
              "threads": 1, "python": platform.python_version(),
              "numpy": np.__version__, "implementations": {}}

    def record(name, index, search, build_seconds, extra=None):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        entry = {"build_seconds": build_seconds, "peak_rss_gib_after_build": rss,
                 "points": [], **(extra or {})}
        for ef in a.ef_search:
            ids, ms = timed_queries(search(index, ef), queries)
            point = {"ef_search": ef, **summarize(ids, ms, truth, a.k)}
            entry["points"].append(point)
            print(f"  {name:<10} ef={ef:<4} recall@10={point['recall_at_10']*100:6.2f}% "
                  f"p50={point['p50_ms']:7.3f}ms p95={point['p95_ms']:7.3f}ms", flush=True)
        report["implementations"][name] = entry

    print("\nBuilding ours (C++)...", flush=True)
    started = time.perf_counter()
    ours = build_ours(vectors, candidates, a.m, a.ef_construction, a.seed)
    ours_seconds = time.perf_counter() - started
    record("ours", ours, lambda i, ef: (lambda q: i.search(q, k=a.k, ef_search=ef)[0]),
           ours_seconds)
    del ours

    print("\nBuilding hnswlib...", flush=True)
    started = time.perf_counter()
    lib = build_hnswlib(vectors, candidates, a.m, a.ef_construction, a.seed)
    lib_seconds = time.perf_counter() - started

    def hnswlib_search(index, ef):
        index.set_ef(ef)
        return lambda q: index.knn_query(q, k=a.k, num_threads=1)[0][0]

    record("hnswlib", lib, hnswlib_search, lib_seconds)
    del lib

    print("\nBuilding faiss...", flush=True)
    started = time.perf_counter()
    fai = build_faiss(vectors, a.m, a.ef_construction)
    faiss_seconds = time.perf_counter() - started

    def faiss_search(index, ef):
        index.hnsw.efSearch = ef
        # FAISS returns positions into its own array; map back to our row IDs.
        return lambda q: candidates[index.search(q.reshape(1, -1), a.k)[1][0]]

    record("faiss", fai, faiss_search, faiss_seconds)
    del fai

    print("\nOur saved graph, for comparison with the fresh build:", flush=True)
    saved = NativeHNSW.load_directory(a.graph)
    if saved.size == len(vectors):
        record("ours_saved", saved,
               lambda i, ef: (lambda q: i.search(q, k=a.k, ef_search=ef)[0]), 0.0,
               {"note": "loaded from disk, not built here; build_seconds is not measured"})
    else:
        print(f"  skipped: saved graph has {saved.size:,} nodes, "
              f"this run used {len(vectors):,}", flush=True)

    ours_p50 = {p["ef_search"]: p["p50_ms"] for p in report["implementations"]["ours"]["points"]}
    print(f"\n{'ef':>5}  {'ours':>9}{'hnswlib':>10}{'faiss':>9}   gap vs hnswlib")
    for ef in a.ef_search:
        h = next(p for p in report["implementations"]["hnswlib"]["points"] if p["ef_search"] == ef)
        f = next(p for p in report["implementations"]["faiss"]["points"] if p["ef_search"] == ef)
        print(f"{ef:>5}  {ours_p50[ef]:>8.3f}ms{h['p50_ms']:>9.3f}ms{f['p50_ms']:>8.3f}ms"
              f"   {ours_p50[ef]/h['p50_ms']:>6.2f}x")
    report["build_seconds"] = {k: v["build_seconds"] for k, v in report["implementations"].items()}
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
