"""Complete the spec's measurement table: recall@100 and multithreaded throughput.

    USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 tests/scaling_table.py \
      --graph store/hnsw_construction_large/cpp_direct \
      --oracle store/large/index_groundtruth.npz --store store/large/vectors.npy

Both rows reuse the existing directly-loaded graph; nothing is rebuilt.

recall@100 needs efSearch >= 100 to be meaningful — a beam narrower than k
cannot return k good neighbours — so the low settings used for recall@10 are
skipped here.

Throughput asks whether search parallelizes or contends. The C++ search releases
the GIL, so Python threads should scale; immutable read-only snapshots are the
easy case for that. It is measured rather than assumed: each thread runs its own
slice of the query set and the aggregate rate is compared against the
single-thread rate. Queries are pinned to the same total count at every thread
count so the comparison is like for like.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import argparse
import json
import platform
import threading
import time

import numpy as np

from timbre.groundtruth import recall_at_k
from timbre.native_hnsw import NativeHNSW


def measure_recall(index, queries, truth, k, ef_search):
    ids, milliseconds = [], []
    for q in queries[:50]:
        index.search(q, k=k, ef_search=ef_search)
    for q in queries:
        started = time.perf_counter()
        found, _ = index.search(q, k=k, ef_search=ef_search)
        milliseconds.append((time.perf_counter() - started) * 1000)
        ids.append(found)
    ids = np.array(ids)
    point = {
        "ef_search": ef_search, "k": k,
        "recall_at_10": float(recall_at_k(ids, truth, min(10, k))),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p99_ms": float(np.percentile(milliseconds, 99)),
    }
    if k >= 100:
        point["recall_at_100"] = float(recall_at_k(ids, truth, 100))
    return point


def measure_throughput(index, queries, k, ef_search, threads, repeats=3):
    """Aggregate queries/sec with `threads` workers over the same total work."""
    for q in queries[:100]:
        index.search(q, k=k, ef_search=ef_search)

    best = 0.0
    for _ in range(repeats):
        slices = np.array_split(np.arange(len(queries)), threads)
        barrier = threading.Barrier(threads + 1)

        def worker(rows):
            barrier.wait()                       # start together, not staggered
            for i in rows:
                index.search(queries[i], k=k, ef_search=ef_search)

        workers = [threading.Thread(target=worker, args=(s,)) for s in slices]
        for w in workers:
            w.start()
        barrier.wait()
        started = time.perf_counter()
        for w in workers:
            w.join()
        elapsed = time.perf_counter() - started
        best = max(best, len(queries) / elapsed)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--ef-search-100", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--throughput-ef", type=int, default=64)
    ap.add_argument("--out", default="docs/scaling_table_results.json")
    a = ap.parse_args()
    if min(a.ef_search_100) < 100:
        ap.error("recall@100 requires efSearch >= 100")

    with np.load(a.oracle, allow_pickle=False) as arrays:
        query_rows = np.asarray(arrays["query_rows"], dtype=np.int64)
        truth = np.asarray(arrays["gt_ids"])
    if truth.shape[1] < 100:
        raise SystemExit("oracle caches fewer than 100 neighbours; cannot score recall@100")
    store = np.load(a.store, mmap_mode="r")
    queries = np.array(store[query_rows])
    del store

    index = NativeHNSW.load_directory(a.graph)
    print(f"{index.size:,} nodes, {len(queries)} held-out queries, "
          f"{os.cpu_count()} logical CPUs\n", flush=True)

    report = {"graph": a.graph, "nodes": int(index.size), "queries": len(queries),
              "python": platform.python_version(), "cpu_count": os.cpu_count(),
              "recall_at_100": [], "throughput": []}

    print(f"{'ef':>5}{'recall@10':>12}{'recall@100':>13}{'p50':>10}{'p95':>10}")
    for ef in a.ef_search_100:
        point = measure_recall(index, queries, truth, 100, ef)
        report["recall_at_100"].append(point)
        print(f"{ef:>5}{point['recall_at_10']*100:>11.2f}%"
              f"{point['recall_at_100']*100:>12.2f}%"
              f"{point['p50_ms']:>9.3f}ms{point['p95_ms']:>9.3f}ms", flush=True)

    print(f"\nThroughput at k=10, efSearch={a.throughput_ef}, best of 3:")
    print(f"{'threads':>8}{'queries/sec':>14}{'speedup':>10}{'efficiency':>12}")
    single = None
    for threads in a.threads:
        qps = measure_throughput(index, queries, 10, a.throughput_ef, threads)
        single = single or qps
        entry = {"threads": threads, "queries_per_second": qps,
                 "speedup": qps / single, "efficiency": qps / single / threads}
        report["throughput"].append(entry)
        print(f"{threads:>8}{qps:>14,.0f}{entry['speedup']:>9.2f}x"
              f"{entry['efficiency']*100:>11.0f}%", flush=True)

    top = max(report["throughput"], key=lambda e: e["queries_per_second"])
    print(f"\nPeak {top['queries_per_second']:,.0f} queries/sec at "
          f"{top['threads']} threads ({top['speedup']:.2f}x over one).")
    if top["efficiency"] < 0.5 and top["threads"] > 1:
        print("Scaling efficiency is under 50%: search contends rather than "
              "parallelizing cleanly at this thread count.")
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
