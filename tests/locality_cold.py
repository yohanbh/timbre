"""Compare two graph layouts at the memory cliff, cold and warm.

    USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 tests/locality_cold.py \
      --graphs store/hnsw_construction_large/cpp_direct store/hnsw_locality_large \
      --labels insertion bfs \
      --oracle store/large/index_groundtruth.npz --store store/large/vectors.npy

The large store measured a 171x p99 penalty on a cold page cache at unchanged
recall. Locality reordering is meant to recover part of that by putting nodes
reached together in a traversal close together on disk.

Cold state is produced with posix_fadvise(POSIX_FADV_DONTNEED) over every file
in the graph directory, which needs no root and no drop_caches. madvise on the
mapped range does not work for this: it returns pages from page cache with zero
major faults. Each layout is measured cold then warm in the same process, and
major faults are reported alongside latency so a "cold" run that never faulted
is visible rather than silently reported as a speedup.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import argparse
import ctypes
import json
import resource
import time

import numpy as np

from timbre.groundtruth import recall_at_k
from timbre.native_hnsw import NativeHNSW

POSIX_FADV_DONTNEED = 4


def evict(directory):
    """Drop every page of the graph's files from the page cache."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def faults():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_minflt, usage.ru_majflt


def run(index, queries, truth, k, ef):
    before = faults()
    ids, milliseconds = [], []
    for query in queries:
        started = time.perf_counter()
        found, _ = index.search(query, k=k, ef_search=ef)
        milliseconds.append((time.perf_counter() - started) * 1000)
        ids.append(found)
    after = faults()
    return {
        "recall_at_10": float(recall_at_k(np.array(ids), truth, k)),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p99_ms": float(np.percentile(milliseconds, 99)),
        "minor_faults": after[0] - before[0],
        "major_faults": after[1] - before[1],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="docs/hnsw_locality_cold_results.json")
    a = ap.parse_args()
    if len(a.graphs) != len(a.labels):
        ap.error("--graphs and --labels must have the same length")

    with np.load(a.oracle, allow_pickle=False) as arrays:
        query_rows = np.asarray(arrays["query_rows"], dtype=np.int64)
        truth = np.asarray(arrays["gt_ids"])
    store = np.load(a.store, mmap_mode="r")
    queries = np.array(store[query_rows])
    del store

    report = {"oracle": a.oracle, "ef_search": a.ef_search, "k": a.k,
              "queries": len(queries), "layouts": {}}
    print(f"{len(queries)} held-out queries at efSearch={a.ef_search}\n")
    print(f"{'layout':<12}{'state':<8}{'recall':>9}{'p50':>10}{'p95':>11}"
          f"{'p99':>12}{'major':>9}")
    for label, path in zip(a.labels, a.graphs):
        index = NativeHNSW.load_directory(path)
        results = {}
        evict(path)
        results["cold"] = run(index, queries, truth, a.k, a.ef_search)
        for _ in range(2):                      # warm every page before timing
            for q in queries[:200]:
                index.search(q, k=a.k, ef_search=a.ef_search)
        results["warm"] = run(index, queries, truth, a.k, a.ef_search)
        for state in ("cold", "warm"):
            r = results[state]
            print(f"{label:<12}{state:<8}{r['recall_at_10']*100:>8.2f}%"
                  f"{r['p50_ms']:>9.3f}ms{r['p95_ms']:>10.3f}ms"
                  f"{r['p99_ms']:>11.3f}ms{r['major_faults']:>9,}")
        results["cold_p99_penalty"] = (results["cold"]["p99_ms"]
                                       / results["warm"]["p99_ms"])
        report["layouts"][label] = results
        del index

    base = a.labels[0]
    print(f"\ncold p99 penalty versus each layout's own warm run:")
    for label in a.labels:
        print(f"  {label:<12}{report['layouts'][label]['cold_p99_penalty']:>8.0f}x")
    if len(a.labels) > 1:
        other = a.labels[1]
        ratio = (report["layouts"][base]["cold"]["p99_ms"]
                 / report["layouts"][other]["cold"]["p99_ms"])
        report["cold_p99_ratio_base_over_other"] = ratio
        print(f"\ncold p99 {base} / {other}: {ratio:.2f}x "
              f"({'reordering helps' if ratio > 1.1 else 'no meaningful recovery'})")
        if any(report["layouts"][l]["cold"]["major_faults"] < 100 for l in a.labels):
            print("WARNING: a cold run took under 100 major faults; eviction "
                  "may not have worked and its timing is not a cold measurement.")
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
