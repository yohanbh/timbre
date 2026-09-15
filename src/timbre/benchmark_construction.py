"""Build Python/C++ HNSW on an existing benchmark's candidates and exact oracle.

    PYTHONPATH=src python3 -m timbre.benchmark_construction

Fresh builds use identical insertion order, layer seed and checkpoint frequency.
All graphs are scored using the native search backend to isolate construction.
"""
from .benchmark_hnsw import measure, prepare  # Pin BLAS threads before NumPy.

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import tempfile
import time

import numpy as np

from . import _hnsw_native
from .groundtruth import load_groundtruth, load_layout, source_identity
from .hnsw import HNSW
from .native_hnsw import NativeHNSW, NativeHNSWBuilder


def _stage_vectors(directory, store, candidates, source):
    """Copy candidate rows once, in bounded chunks, into construction order."""
    directory = Path(directory)
    identity = {
        "candidate_rows_sha256": hashlib.sha256(candidates.tobytes()).hexdigest(),
        "source_store_sha256": str(source["source_store_sha256"]),
        "shape": [len(candidates), store.shape[1]],
        "dtype": np.dtype(np.float32).str,
    }
    metadata_path = directory / "metadata.json"
    if directory.exists():
        if not metadata_path.is_file() or json.loads(metadata_path.read_text()) != identity:
            raise ValueError("staged candidate vectors do not match this benchmark")
        return np.load(directory / "vectors.npy", mmap_mode="r")

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        vectors = np.lib.format.open_memmap(
            temporary / "vectors.npy", mode="w+", dtype=np.float32,
            shape=(len(candidates), store.shape[1]),
        )
        for start in range(0, len(candidates), 8192):
            stop = min(start + 8192, len(candidates))
            vectors[start:stop] = store[candidates[start:stop]]
        vectors.flush()
        del vectors
        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return np.load(directory / "vectors.npy", mmap_mode="r")


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
    ap.add_argument("--out", type=Path, default=Path("store/hnsw_construction_medium"))
    ap.add_argument("--backends", nargs="+", choices=["python", "cpp"], default=["python", "cpp"])
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--groundtruth", default="store/groundtruth.npz")
    ap.add_argument("--oracle", type=Path, default=None,
                    help="Benchmark the provided held-out oracle directly.")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--ef-construction", type=int, default=80)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--k", type=int, default=10,
                    help="neighbors requested per query and used for recall scoring")
    a = ap.parse_args()
    if a.m < 2 or a.ef_construction < a.m or min(a.ef_search) < a.k or a.k < 10:
        ap.error("require M >= 2, efConstruction >= M, k >= 10, and efSearch >= k")
    if a.out.resolve() == a.baseline.resolve():
        ap.error("choose a separate output directory from the baseline")
    baseline = json.loads((a.baseline / "results.json").read_text())
    oracle = a.oracle or (a.baseline / "groundtruth.npz")
    if not oracle.exists():
        ap.error("baseline exact oracle is missing; run benchmark_hnsw first")
    store = np.load(a.store, mmap_mode="r")
    rows, owners = load_layout(a.db)
    if a.oracle is None:
        phase1 = load_groundtruth(a.groundtruth, a.db, a.store)
        if baseline["queries"] > len(phase1["query_rows"]):
            ap.error("requested more queries than the fixed Phase 1 query set")
        candidates, query_rows, truth = prepare(
            store, rows, owners, phase1, baseline["vectors"], baseline["queries"],
            baseline["seed"], oracle,
        )
        source = phase1
    else:
        candidates, query_rows, truth, source = _load_oracle(oracle, a.db, a.store)
        if not np.intersect1d(candidates, query_rows).size == 0:
            raise ValueError("oracle candidates overlap held-out queries")
        if not np.in1d(candidates, rows, assume_unique=True).all():
            raise ValueError("oracle candidates are not a subset of embedded rows")
        if "seed" in source and int(source["seed"]) != baseline["seed"]:
            raise ValueError(f"oracle seed {source['seed']} does not match baseline seed {baseline['seed']}")
        print(f"Using oracle candidates={len(candidates):,} queries={len(query_rows):,}", flush=True)
        a.vectors, a.queries = len(candidates), len(query_rows)
    a.out.mkdir(parents=True, exist_ok=True)
    vectors = _stage_vectors(a.out / "candidate_vectors", store, candidates, source)
    queries = np.array(store[query_rows])
    del store
    order = np.arange(len(vectors))
    if baseline["order"] == "shuffled":
        order = np.random.default_rng(baseline["seed"] + 1).permutation(order)
    identity = {
        "vectors": len(vectors), "queries": len(queries), "dimension": vectors.shape[1],
        "M": a.m, "ef_construction": a.ef_construction, "seed": baseline["seed"],
        "order": baseline["order"], "blas_threads": 1,
        "source_model": str(source["source_model"]),
        "source_store_sha256": str(source["source_store_sha256"]),
        "oracle_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
        "native_backend": NativeHNSWBuilder.BACKEND,
    }
    report = {**identity, "python_version": platform.python_version(), "numpy_version": np.__version__,
              "compiler": _hnsw_native.compiler, "pybind11_version": _hnsw_native.pybind11_version,
              "compile_flags": "-O3 -ffp-contract=off; C++17", "query_backend": "cpp",
              "checkpoint_every": 25_000, "build_repetitions": 1, "builds": []}
    identity_path = a.out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("benchmark source/configuration changed; choose a new --out directory")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    for backend in a.backends:
        path, info_path = a.out / f"{backend}.npz", a.out / f"{backend}.json"
        cls = HNSW if backend == "python" else NativeHNSWBuilder
        gc.collect()
        started = time.perf_counter()
        index = cls.load(path, vectors) if path.exists() else cls(
            vectors, candidates, M=a.m, ef_construction=a.ef_construction, seed=baseline["seed"]
        )
        initialization_seconds = time.perf_counter() - started
        if (index.M != a.m or index.ef_construction != a.ef_construction or
                not np.array_equal(index.ids, candidates)):
            raise ValueError("checkpoint does not match benchmark configuration/IDs")
        resumed = index.size > 0
        if index.size < len(vectors):
            checkpoint_seconds = 0.0

            def progress(n):
                nonlocal checkpoint_seconds
                if n % 25_000 == 0:
                    saving = time.perf_counter()
                    index.save(path)
                    checkpoint_seconds += time.perf_counter() - saving
                if n % 10_000 == 0:
                    print(f"  {backend}: {n:,}/{len(vectors):,}; {time.perf_counter() - started:.1f}s", flush=True)

            print(f"Building {backend}: {index.size:,}/{len(vectors):,} already inserted", flush=True)
            insertion_started = time.perf_counter()
            index.build(order[index.levels[order] < 0], progress)
            insertion_seconds = time.perf_counter() - insertion_started - checkpoint_seconds
            saving = time.perf_counter()
            index.save(path)
            final_save_seconds = time.perf_counter() - saving
            info = {"resumed": resumed, "build_seconds": None if resumed else time.perf_counter() - started,
                    "initialization_seconds": initialization_seconds, "insertion_seconds": insertion_seconds,
                    "checkpoint_seconds": checkpoint_seconds, "final_save_seconds": final_save_seconds}
            info_path.write_text(json.dumps(info, indent=2) + "\n")
        else:
            print(f"Loaded completed {backend} build", flush=True)
        info = json.loads(info_path.read_text()) if info_path.exists() else {"build_seconds": None}
        levels_sha = hashlib.sha256(index.levels.tobytes()).hexdigest()
        native = NativeHNSW(index) if backend == "python" else index.freeze()
        del index
        gc.collect()
        build = {"backend": backend, **info, "graph_bytes": path.stat().st_size,
                 "graph_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "levels_sha256": levels_sha, "points": []}
        if backend == "cpp":
            direct_path = a.out / "cpp_direct"
            if not direct_path.exists():
                native.save_directory(direct_path)
            build["direct_graph"] = str(direct_path)
            build["direct_graph_bytes"] = sum(p.stat().st_size for p in direct_path.iterdir())
        print(f"{backend} total build: {info['build_seconds']}s", flush=True)
        for ef in a.ef_search:
            point = measure(native, queries, truth, ef, a.k)
            build["points"].append(point)
            recall_100 = (
                f", recall@100={point['recall_at_100']:.4f}"
                if "recall_at_100" in point else ""
            )
            print(f"  efSearch={ef:3d}: recall@10={point['recall_at_10']:.4f}{recall_100}, "
                  f"p50={point['p50_ms']:.3f}ms", flush=True)
        report["builds"].append(build)
        report["passes_recall_gate"] = any(p["recall_at_10"] >= 0.95
                                           for b in report["builds"] for p in b["points"])
        if {b["backend"] for b in report["builds"]} == {"python", "cpp"}:
            py, cpp = (next(b for b in report["builds"] if b["backend"] == name) for name in ("python", "cpp"))
            if py["levels_sha256"] != cpp["levels_sha256"]:
                raise RuntimeError("construction backends assigned different node levels")
            if py["build_seconds"] and cpp["build_seconds"]:
                report["total_build_speedup"] = py["build_seconds"] / cpp["build_seconds"]
                report["insertion_speedup"] = py["insertion_seconds"] / cpp["insertion_seconds"]
        report["comparison_process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (a.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        del native
    print(f"Wrote {a.out / 'results.json'}", flush=True)
    return 0 if report["passes_recall_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
