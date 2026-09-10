"""Phase 2: reproducible HNSW sweep against exact held-out segment neighbors.

    PYTHONPATH=src python3 -m timbre.benchmark_hnsw

The Phase 2 default is 125k indexed vectors and the existing 1,000 queries.
All query rows are excluded from construction. Exact ground truth is recomputed
over this precise candidate set; the full-corpus Phase 1 cache is not a valid
oracle for a smaller index. Outputs are kept separate from the production store.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("USE_TF", "0")

import argparse
import gc
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from .groundtruth import exact_topk, load_groundtruth, load_layout, owner_lookup, recall_at_k
from .hnsw import HNSW


def prepare(store, row_ids, owner, phase1, n_vectors, n_queries, seed, path):
    """Freeze candidate/query membership before computing exact neighbors."""
    rng = np.random.default_rng(seed)
    queries = phase1["query_rows"]
    if n_queries < len(queries):
        queries = np.sort(rng.choice(queries, n_queries, replace=False))
    candidates = np.setdiff1d(row_ids, queries, assume_unique=True)
    if n_vectors > len(candidates):
        raise ValueError(f"only {len(candidates)} vectors remain after query holdout")
    candidates = np.sort(rng.choice(candidates, n_vectors, replace=False))
    identity = {key: str(value) for key, value in phase1.items() if key.startswith("source_")}
    metadata = json.dumps({**identity, "seed": seed, "blas_threads": 1}, sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if (str(data["metadata"]) != metadata or
                    not np.array_equal(data["candidate_rows"], candidates) or
                    not np.array_equal(data["query_rows"], queries)):
                raise ValueError("benchmark source/sample changed; choose a new --out directory")
            return candidates, queries, data["gt_ids"].copy()

    print(f"Exact oracle: {len(queries)} held-out queries × {len(candidates)} candidates", flush=True)
    ids, scores = exact_topk(store, queries, candidates, owner_lookup(row_ids, owner), k=100)
    np.savez(path, candidate_rows=candidates, query_rows=queries, gt_ids=ids,
             gt_sims=scores, metadata=metadata)
    return candidates, queries, ids


def frontier(points):
    """Points not dominated in both recall@10 and median query latency."""
    return [p for p in points if not any(
        other["recall_at_10"] >= p["recall_at_10"] and other["p50_ms"] <= p["p50_ms"] and
        (other["recall_at_10"] > p["recall_at_10"] or other["p50_ms"] < p["p50_ms"])
        for other in points
    )]


def measure(index, queries, truth, ef_search):
    # Warm caches without including model load, construction or disk IO in timing.
    for query in queries[:10]:
        index.search(query, k=10, ef_search=ef_search)
    ids, milliseconds, evaluations = [], [], []
    for query in queries:
        started = time.perf_counter()
        found, _ = index.search(query, k=10, ef_search=ef_search)
        milliseconds.append((time.perf_counter() - started) * 1000)
        evaluations.append(index.distance_evaluations)
        if len(found) != 10:
            raise RuntimeError("graph returned fewer than ten neighbors")
        ids.append(found)
    ids = np.array(ids)
    return {
        "ef_search": ef_search,
        "recall_at_1": recall_at_k(ids, truth, 1),
        "recall_at_10": recall_at_k(ids, truth, 10),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p99_ms": float(np.percentile(milliseconds, 99)),
        "queries_per_second": float(1000 / np.mean(milliseconds)),
        "mean_distance_evaluations": float(np.mean(evaluations)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--groundtruth", default="store/groundtruth.npz")
    ap.add_argument("--out", type=Path, default=Path("store/hnsw_phase2"))
    ap.add_argument("--vectors", type=int, default=125_000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--m", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--ef-construction", type=int, nargs="+", default=[80, 160])
    ap.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--order", choices=["shuffled", "track"], default="shuffled")
    a = ap.parse_args()
    if (a.vectors < 100 or a.queries < 1 or min(a.m) < 2 or
            min(a.ef_construction) < max(a.m) or min(a.ef_search) < 10):
        ap.error("require vectors >= 100, queries >= 1, M >= 2, efConstruction >= M, efSearch >= 10")

    phase1 = load_groundtruth(a.groundtruth, a.db, a.store)
    if a.queries > len(phase1["query_rows"]):
        ap.error("requested more queries than the fixed Phase 1 query set")
    a.out.mkdir(parents=True, exist_ok=True)
    store = np.load(a.store, mmap_mode="r")
    row_ids, owner = load_layout(a.db)
    candidates, query_rows, truth = prepare(
        store, row_ids, owner, phase1, a.vectors, a.queries, a.seed, a.out / "groundtruth.npz"
    )
    vectors, queries = np.array(store[candidates]), np.array(store[query_rows])
    order = np.arange(len(vectors))
    if a.order == "shuffled":
        order = np.random.default_rng(a.seed + 1).permutation(order)
    report = {"vectors": len(vectors), "queries": len(queries), "dimension": vectors.shape[1],
              "seed": a.seed, "order": a.order, "blas_threads": 1,
              "numpy_version": np.__version__, "python_version": platform.python_version(),
              "source_model": str(phase1["source_model"]),
              "source_store_sha256": str(phase1["source_store_sha256"]), "points": []}

    for M in a.m:
        for ef_construction in a.ef_construction:
            label = f"M{M}_efC{ef_construction}_{a.order}_seed{a.seed}"
            path = a.out / f"{label}.npz"
            build_info = a.out / f"{label}.json"
            started = time.perf_counter()
            index = HNSW.load(path, vectors) if path.exists() else HNSW(
                vectors, candidates, M=M, ef_construction=ef_construction, seed=a.seed
            )
            if index.M != M or index.ef_construction != ef_construction or not np.array_equal(index.ids, candidates):
                raise ValueError("saved graph configuration/IDs do not match benchmark")
            resumed = index.size > 0
            if index.size < len(vectors):
                print(f"Building {label}: {index.size}/{len(vectors)} already inserted", flush=True)

                def progress(n):
                    if n % 10_000 == 0:
                        print(f"  {n}/{len(vectors)}  elapsed {time.perf_counter() - started:.1f}s", flush=True)
                    if n % 25_000 == 0:
                        index.save(path)

                index.build(order[index.levels[order] < 0], progress)
                build_seconds = time.perf_counter() - started if not resumed else None
                index.save(path)
                build_info.write_text(json.dumps({"build_seconds": build_seconds,
                                                  "resumed": resumed}, indent=2) + "\n")
            else:
                print(f"Loaded {label}", flush=True)
            info = json.loads(build_info.read_text()) if build_info.exists() else {"build_seconds": None}
            for ef_search in a.ef_search:
                point = {"M": M, "ef_construction": ef_construction, **info,
                         "graph_bytes": path.stat().st_size,
                         **measure(index, queries, truth, ef_search)}
                report["points"].append(point)
                print(f"  efSearch={ef_search:3d}: recall@10={point['recall_at_10']:.4f} "
                      f"p50={point['p50_ms']:.3f}ms p95={point['p95_ms']:.3f}ms", flush=True)
                report["frontier"] = frontier(report["points"])
                report["passes_recall_gate"] = any(p["recall_at_10"] >= 0.95 for p in report["points"])
                # Linux ru_maxrss is KiB; this is the whole benchmark process,
                # including imports and source arrays, not just the graph.
                report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                (a.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            del index
            gc.collect()
    print(f"Wrote {a.out / 'results.json'}; recall@10 gate: {report['passes_recall_gate']}", flush=True)
    return 0 if report["passes_recall_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
