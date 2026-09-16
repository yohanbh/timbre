"""Reorder a direct graph by traversal locality and measure what it recovers.

    PYTHONPATH=src python3 -m timbre.reorder \
      --graph store/hnsw_construction_large/cpp_direct \
      --out store/hnsw_locality_large \
      --oracle store/large/index_groundtruth.npz \
      --store store/large/vectors.npy

HNSW search is a pointer chase. Once the graph outgrows RAM every hop is a
potential page fault, and the large store measured a 171x p99 penalty on a cold
page cache (173.749 ms versus 1.015 ms, 1,022 major faults versus zero) at
unchanged recall. Nodes are currently laid out in insertion order, which is
shuffled, so neighbours are scattered across the whole mapping.

This permutes node IDs into BFS order from the entrypoint, so nodes reached
close together in a traversal sit close together on disk and share pages. The
permutation is a pure relabelling: every vector, level and edge survives, so
recall must be identical afterwards. That invariant is checked rather than
assumed -- a reordering that changes recall has corrupted the graph.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .native_hnsw import NativeHNSW


def bfs_order(index):
    """Node ids in breadth-first order from the entrypoint, layer 0 edges.

    Layer 0 holds every node and carries the hops that dominate a query, so it
    is the layer whose locality matters. Nodes unreachable from the entrypoint
    (none in a well-formed graph, but the check is cheap) keep their relative
    order at the end so the permutation stays a bijection.
    """
    nodes, offsets, links = index._layers[0]
    position = np.full(len(index.levels), -1, dtype=np.int64)
    position[np.asarray(nodes, dtype=np.int64)] = np.arange(len(nodes))

    seen = np.zeros(len(index.levels), dtype=bool)
    order = np.empty(len(nodes), dtype=np.int64)
    filled = 0
    queue = deque([int(index.entrypoint)])
    seen[index.entrypoint] = True
    while queue:
        node = queue.popleft()
        order[filled] = node
        filled += 1
        p = position[node]
        if p < 0:
            continue
        for neighbour in links[offsets[p]:offsets[p + 1]]:
            n = int(neighbour)
            if not seen[n]:
                seen[n] = True
                queue.append(n)
    if filled < len(nodes):
        rest = [int(n) for n in nodes if not seen[n]]
        order[filled:] = rest
        filled += len(rest)
    if filled != len(nodes):
        raise RuntimeError("BFS did not cover every layer-0 node exactly once")
    return order


def permute(index, order, path):
    """Write a relabelled copy of `index` whose new id i is old id order[i]."""
    new_of_old = np.empty(len(index.levels), dtype=np.int64)
    new_of_old[order] = np.arange(len(order))

    snapshot = NativeHNSW.__new__(NativeHNSW)
    snapshot.vectors = np.ascontiguousarray(np.asarray(index.vectors)[order])
    snapshot.ids = np.asarray(index.ids)[order].copy()
    snapshot.levels = np.asarray(index.levels)[order].copy()
    snapshot.M = index.M
    snapshot.ef_construction = index.ef_construction
    snapshot.size = index.size
    snapshot.max_level = index.max_level
    snapshot.entrypoint = int(new_of_old[index.entrypoint])
    snapshot.construction_backend = index.construction_backend

    layers = []
    for nodes, offsets, links in index._layers:
        old_nodes = np.asarray(nodes, dtype=np.int64)
        relabelled = new_of_old[old_nodes]
        rank = np.argsort(relabelled, kind="stable")      # keep layers id-sorted
        counts = np.diff(np.asarray(offsets))[rank]
        new_offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        new_links = np.empty(len(links), dtype=np.int32)
        at = 0
        old_offsets = np.asarray(offsets)
        for r in rank:
            block = links[old_offsets[r]:old_offsets[r + 1]]
            new_links[at:at + len(block)] = new_of_old[np.asarray(block, dtype=np.int64)]
            at += len(block)
        layers.append((relabelled[rank].astype(np.int32), new_offsets, new_links))
    snapshot._layers = layers

    from ._hnsw_native import SearchIndex
    snapshot._index = SearchIndex(snapshot.vectors, snapshot.ids,
                                  snapshot.entrypoint, layers)
    snapshot.distance_evaluations = 0
    snapshot.save_directory(path)
    return snapshot


def page_locality(index, page_bytes=4096):
    """Mean distinct vector pages touched by one node's layer-0 neighbourhood.

    This is the quantity reordering is meant to shrink, measured directly on the
    layout rather than inferred from timings.
    """
    nodes, offsets, links = index._layers[0]
    per_vector = index.vectors.shape[1] * index.vectors.dtype.itemsize
    rng = np.random.default_rng(0)
    sample = rng.choice(len(nodes), size=min(20000, len(nodes)), replace=False)
    total = 0
    offsets = np.asarray(offsets)
    for p in sample:
        block = np.asarray(links[offsets[p]:offsets[p + 1]], dtype=np.int64)
        total += len(np.unique(block * per_vector // page_bytes))
    return total / len(sample)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True, help="source direct-graph directory")
    ap.add_argument("--out", required=True, help="destination for the reordered graph")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--report", default="docs/hnsw_locality_results.json")
    ap.add_argument("--reuse", action="store_true",
                    help="verify an already-written --out graph instead of permuting again")
    a = ap.parse_args()

    from .groundtruth import recall_at_k

    print(f"Loading {a.graph}...", flush=True)
    index = NativeHNSW.load_directory(a.graph)
    with np.load(a.oracle, allow_pickle=False) as arrays:
        query_rows = np.asarray(arrays["query_rows"], dtype=np.int64)
        truth = np.asarray(arrays["gt_ids"])
    store = np.load(a.store, mmap_mode="r")
    queries = np.array(store[query_rows])
    del store
    print(f"{index.size:,} nodes, {len(queries)} held-out queries", flush=True)

    before = page_locality(index)
    if a.reuse:
        bfs_seconds = permute_seconds = 0.0
        reordered = NativeHNSW.load_directory(a.out)
        if reordered.size != index.size:
            raise SystemExit("--reuse graph does not match the source graph")
        print(f"Reusing existing {a.out}", flush=True)
    else:
        started = time.perf_counter()
        order = bfs_order(index)
        bfs_seconds = time.perf_counter() - started
        print(f"BFS order computed in {bfs_seconds:.1f}s", flush=True)

        started = time.perf_counter()
        reordered = permute(index, order, Path(a.out))
        permute_seconds = time.perf_counter() - started
        print(f"Permuted and written in {permute_seconds:.1f}s", flush=True)
    after = page_locality(reordered)
    print(f"vector pages per neighbourhood: {before:.2f} -> {after:.2f}", flush=True)

    # Recall must be unchanged: a permutation relabels, it does not rewire.
    report = {"graph": a.graph, "out": a.out, "nodes": int(index.size),
              "queries": len(queries), "k": a.k,
              "bfs_seconds": bfs_seconds, "permute_seconds": permute_seconds,
              "pages_per_neighbourhood": {"before": before, "after": after},
              "points": []}
    # Finish one graph entirely before opening the other. Interleaving them
    # keeps two multi-gigabyte mappings hot at once, which thrashes whenever
    # they do not both fit in RAM: an interleaved 2.14M run took 607,357 major
    # faults and 1.3 TB of reads before being abandoned.
    def sweep(graph):
        return {ef: np.array([graph.search(q, k=a.k, ef_search=ef)[0] for q in queries])
                for ef in a.ef_search}

    print("\nSearching the original layout...", flush=True)
    original = sweep(index)
    del index
    print("Searching the reordered layout...", flush=True)
    moved = sweep(reordered)

    print(f"\n{'ef':>4}{'recall before':>15}{'recall after':>14}")
    for ef in a.ef_search:
        a_ids, b_ids = original[ef], moved[ef]
        ra = float(recall_at_k(a_ids, truth, a.k))
        rb = float(recall_at_k(b_ids, truth, a.k))
        same = bool(np.array_equal(np.sort(a_ids, axis=1), np.sort(b_ids, axis=1)))
        print(f"{ef:>4}{ra*100:>14.2f}%{rb*100:>13.2f}%"
              f"{'   identical sets' if same else '   SETS DIFFER'}")
        report["points"].append({"ef_search": ef, "recall_before": ra,
                                 "recall_after": rb, "identical_sets": same})
    if any(abs(p["recall_before"] - p["recall_after"]) > 1e-12 for p in report["points"]):
        raise SystemExit("reordering changed recall; the permutation is wrong")
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.report, "w"), indent=2)
    print(f"\nWrote {a.report}")
    print("Measure the cold-cache effect separately: evict both graphs with "
          "posix_fadvise and compare p99.")


if __name__ == "__main__":
    raise SystemExit(main())
