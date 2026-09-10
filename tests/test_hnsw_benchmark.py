"""The recall oracle must use exactly the graph's candidate membership."""
import numpy as np
import pytest

from timbre.benchmark_hnsw import frontier, prepare


def test_benchmark_holds_out_queries_and_recomputes_exact_neighbors(tmp_path):
    rng = np.random.default_rng(5)
    store = rng.normal(size=(160, 512)).astype(np.float32)
    store /= np.linalg.norm(store, axis=1, keepdims=True)
    # Leave holes in the candidate layout, as the real mmap does.
    rows = np.arange(150)
    rows = rows[rows % 11 != 0]
    owner = rows // 5
    phase1 = {"query_rows": np.array([1, 7, 19]), "source_store_sha256": "source-a"}
    path = tmp_path / "truth.npz"
    candidates, queries, truth = prepare(store, rows, owner, phase1, 100, 3, 42, path)
    assert not np.intersect1d(candidates, queries).size
    assert set(candidates) <= set(rows)
    expected = candidates[np.argsort(-(store[queries] @ store[candidates].T), axis=1)]
    np.testing.assert_array_equal(truth, expected)
    cached = prepare(store, rows, owner, phase1, 100, 3, 42, path)
    for a, b in zip((candidates, queries, truth), cached):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="changed"):
        prepare(store, rows, owner, phase1, 101, 3, 42, path)
    phase1["source_store_sha256"] = "source-b"
    with pytest.raises(ValueError, match="changed"):
        prepare(store, rows, owner, phase1, 100, 3, 42, path)


def test_frontier_removes_only_dominated_points():
    points = [{"recall_at_10": r, "p50_ms": t}
              for r, t in [(0.9, 1), (0.95, 2), (0.9, 3), (0.99, 4)]]
    assert frontier(points) == [points[0], points[1], points[3]]
