"""Compare Python and C++ search on the same saved graph and validated oracle.

    PYTHONPATH=src python3 -m timbre.benchmark_native

No embeddings, exact neighbors or graphs are rebuilt. Each latency statistic
is the median of repeated 1,000-query passes, alternating backend order.
"""
from .benchmark_hnsw import measure, prepare  # Pin BLAS threads before NumPy.

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from . import _hnsw_native
from .groundtruth import load_groundtruth, load_layout, recall_at_k, source_identity
from .hnsw import HNSW
from .native_hnsw import NativeHNSW


def _load_oracle(path, db_path, store_path):
    """Load a held-out oracle with candidate/query rows and exact top-k ids."""
    with np.load(path, allow_pickle=False) as arrays:
        oracle = {key: arrays[key].copy() for key in arrays.files}
    required = {"candidate_rows", "query_rows", "gt_ids"}
    if not required.issubset(oracle):
        raise ValueError("oracle is missing candidate_rows/query_rows/gt_ids")
    candidate_rows = np.asarray(oracle["candidate_rows"], dtype=np.int64)
    query_rows = np.asarray(oracle["query_rows"], dtype=np.int64)
    truth = np.asarray(oracle["gt_ids"])
    meta = oracle.get("metadata")
    if meta:
        try:
            parsed = json.loads(str(meta))
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            oracle = {**oracle, **parsed}
    identity = source_identity(db_path, store_path)
    if any(str(oracle.get(key)) != value for key, value in identity.items()):
        raise RuntimeError("held-out oracle is stale; rebuild it for this store")
    return candidate_rows, query_rows, truth, oracle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=Path("store/hnsw_medium"))
    ap.add_argument("--out", type=Path, default=Path("store/hnsw_native_medium/results.json"))
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--groundtruth", default="store/groundtruth.npz")
    ap.add_argument("--oracle", type=Path, default=None,
                    help="Benchmark against this held-out oracle instead of baseline groundtruth.")
    ap.add_argument("--graph", type=Path, default=None,
                    help="NPZ graph or direct graph directory when --oracle is provided.")
    ap.add_argument("--verify-direct", action="store_true",
                    help="Hash every direct graph array before benchmarking.")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--ef-construction", type=int, default=80)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--k", type=int, default=10,
                    help="neighbors requested per query and used for recall scoring")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    if a.repeats < 1 or min(a.ef_search) < a.k or a.k < 10:
        ap.error("require repeats >= 1, k >= 10, and efSearch >= k")
    baseline = json.loads((a.baseline / "results.json").read_text())
    oracle_path = a.oracle or (a.baseline / "groundtruth.npz")
    if not oracle_path.exists():
        ap.error("baseline exact oracle is missing; run benchmark_hnsw first or pass --oracle")
    if a.oracle is not None and a.graph is None:
        ap.error("when using --oracle, pass --graph with the saved C++ or Python benchmark graph")
    label = f"M{a.m}_efC{a.ef_construction}_{baseline['order']}_seed{baseline['seed']}"
    graph_path = a.graph or (a.baseline / f"{label}.npz")
    if a.out.resolve() in {p.resolve() for p in a.baseline.iterdir()}:
        ap.error("--out must not overwrite a baseline artifact")
    store = np.load(a.store, mmap_mode="r")
    row_ids, owner = load_layout(a.db)
    if a.oracle is None:
        phase1 = load_groundtruth(a.groundtruth, a.db, a.store)
        candidates, query_rows, truth = prepare(
            store, row_ids, owner, phase1, baseline["vectors"], baseline["queries"],
            baseline["seed"], oracle_path,
        )
        source = phase1
    else:
        candidates, query_rows, truth, source = _load_oracle(oracle_path, a.db, a.store)
        if not np.intersect1d(candidates, query_rows).size == 0:
            raise ValueError("oracle candidates overlap held-out queries")
        if not np.in1d(candidates, row_ids, assume_unique=True).all():
            raise ValueError("oracle candidates are not a subset of embedded rows")
        if "seed" in source and int(source["seed"]) != baseline["seed"]:
            raise ValueError(f"oracle seed {source['seed']} does not match baseline seed {baseline['seed']}")
        print(f"Benchmarking held-out oracle with {len(candidates):,} candidates and {len(query_rows):,} queries",
              flush=True)
    queries = np.array(store[query_rows])
    direct = graph_path.is_dir()
    vectors = None if direct else store[candidates]
    del store
    gc.collect()
    started = time.perf_counter()
    if direct:
        python = None
        cpp = NativeHNSW.load_directory(graph_path, verify=a.verify_direct)
        vectors = cpp.vectors
    else:
        python = HNSW.load(graph_path, vectors)
        cpp = NativeHNSW(python)
    native_load_seconds = time.perf_counter() - started
    if (cpp.size != len(vectors) or not np.array_equal(cpp.ids, candidates) or
            cpp.M != a.m or cpp.ef_construction != a.ef_construction):
        raise ValueError("saved graph configuration/IDs do not match benchmark")
    if direct:
        graph_bytes = sum(path.stat().st_size for path in graph_path.iterdir())
        graph_sha256 = hashlib.sha256((graph_path / "metadata.json").read_bytes()).hexdigest()
        graph_hash_scope = "metadata manifest (contains per-file SHA-256 fingerprints)"
    else:
        graph_bytes = graph_path.stat().st_size
        graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
        graph_hash_scope = "NPZ file"
    report = {
        "vectors": len(vectors), "queries": len(queries), "dimension": vectors.shape[1],
        "M": a.m, "ef_construction": a.ef_construction, "seed": baseline["seed"],
        "order": baseline["order"], "blas_threads": 1, "repeats": a.repeats,
        "summary": "median of per-pass statistics; backend order alternates each pass",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pybind11_version": _hnsw_native.pybind11_version,
        "compiler": _hnsw_native.compiler, "compile_flags": "-O3 -ffp-contract=off; C++17",
        "platform": platform.platform(),
        "source_model": str(source["source_model"]),
        "source_store_sha256": str(source["source_store_sha256"]),
        "graph_sha256": graph_sha256, "graph_hash_scope": graph_hash_scope,
        "oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        "graph_bytes": graph_bytes,
        "native_load_seconds": native_load_seconds,
        "direct_graph": direct, "direct_checksums_verified": direct and a.verify_direct,
        "construction_backend": cpp.construction_backend or "python", "points": [],
    }
    print(f"Loaded {len(vectors):,} nodes; native load {native_load_seconds:.2f}s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ef in a.ef_search:
        runs = {"cpp": []} if direct else {"python": [], "cpp": []}
        for repeat in range(a.repeats):
            backends = [("cpp", cpp)] if direct else [("python", python), ("cpp", cpp)]
            if not direct and repeat % 2:
                backends.reverse()
            for name, index in backends:
                runs[name].append(measure(index, queries, truth, ef, a.k))
        summary = {name: {key: float(np.median([run[key] for run in passes]))
                          for key in passes[0] if key != "ef_search"}
                   for name, passes in runs.items()}
        point = {"ef_search": ef, **summary, "runs": runs}
        if not direct:
            expected = np.array([python.search(q, k=a.k, ef_search=ef)[0] for q in queries])
            actual = np.array([cpp.search(q, k=a.k, ef_search=ef)[0] for q in queries])
            point.update({
                "p50_speedup": summary["python"]["p50_ms"] / summary["cpp"]["p50_ms"],
                "p95_speedup": summary["python"]["p95_ms"] / summary["cpp"]["p95_ms"],
                "neighbor_overlap_at_10": recall_at_k(actual, expected, min(10, a.k)),
                "identical_ordered_results_fraction": float(np.mean(np.all(actual == expected, axis=1))),
            })
        report["points"].append(point)
        report["passes_recall_gate"] = any(p["cpp"]["recall_at_10"] >= 0.95
                                           for p in report["points"])
        # This includes BOTH backends and loading/validation, not native serving RSS.
        report["comparison_process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        a.out.write_text(json.dumps(report, indent=2) + "\n")
        recall_100 = (
            f", C++ recall@100={summary['cpp'].get('recall_at_100', float('nan')):.4f}"
            if "recall_at_100" in summary["cpp"] else ""
        )
        comparison = (
            f"Python {summary['python']['p50_ms']:.3f}ms, C++ {summary['cpp']['p50_ms']:.3f}ms "
            f"({point['p50_speedup']:.2f}x)"
            if not direct else f"C++ {summary['cpp']['p50_ms']:.3f}ms"
        )
        print(f"efSearch={ef:3d}: {comparison}; "
              f"C++ recall@10={summary['cpp']['recall_at_10']:.4f}{recall_100}", flush=True)
    print(f"Wrote {a.out}", flush=True)
    return 0 if report["passes_recall_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
