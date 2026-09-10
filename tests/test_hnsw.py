"""Behavioral checks for the graph, including held-out recall and persistence."""
import numpy as np
import pytest

from timbre.hnsw import HNSW


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_heuristic_preserves_a_different_direction():
    angles = np.deg2rad([5, 6, -20])
    vectors = unit(np.column_stack([np.cos(angles), np.sin(angles)]))
    index = HNSW(vectors, M=2, ef_construction=4)
    distances = index._distances(np.array([1, 0], dtype=np.float32), [0, 1, 2])
    # The two closest candidates are redundant. Keep a route on the other side.
    assert index._select_neighbors(zip(distances.tolist(), [0, 1, 2]), 2) == [0, 2]


def test_small_graph_connects_both_directions_and_excludes_uninserted():
    index = HNSW(unit([[1, 0], [0.9, 0.1], [0, 1]]), ids=[20, 30, 40], M=4)
    index.insert(0)
    index.insert(1)
    assert index.base[0] == [1]
    assert index.base[1] == [0]
    ids, _ = index.search([0, 1], k=3, ef_search=3)
    assert set(ids) == {20, 30}
    index.validate()
    with pytest.raises(ValueError, match="uninserted"):
        index.insert(0)


def test_hierarchy_degree_bounds_and_exact_when_beam_covers_graph():
    rng = np.random.default_rng(14)
    vectors = unit(rng.normal(size=(250, 16)))
    index = HNSW(vectors, M=8, ef_construction=80, seed=4).build(rng.permutation(250))
    index.validate()
    assert index.max_level >= 1
    assert 5 <= (index.levels >= 1).sum() <= 65
    for query in unit(rng.normal(size=(12, 16))):
        ids, scores = index.search(query, k=10, ef_search=len(vectors))
        expected = np.argsort(-(vectors @ query))[:10]
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_allclose(scores, vectors[ids] @ query, atol=1e-6)


def test_held_out_clustered_queries_have_high_recall():
    rng = np.random.default_rng(9)
    centers = unit(rng.normal(size=(20, 32)))
    vectors = unit(centers[rng.integers(20, size=1000)] + 0.1 * rng.normal(size=(1000, 32)))
    queries = unit(centers[rng.integers(20, size=60)] + 0.1 * rng.normal(size=(60, 32)))
    index = HNSW(vectors, M=12, ef_construction=80, seed=3).build(rng.permutation(len(vectors)))
    truth = np.argsort(-(queries @ vectors.T), axis=1)[:, :10]
    hits = [len(set(index.search(q, k=10, ef_search=80)[0]) & set(t))
            for q, t in zip(queries, truth)]
    assert np.mean(hits) / 10 >= 0.95
    index.validate()


def test_roundtrip_preserves_search_and_continued_insertion(tmp_path):
    rng = np.random.default_rng(1)
    vectors = unit(rng.normal(size=(150, 12)))
    index = HNSW(vectors, ids=np.arange(150) * 21, M=6, ef_construction=40, seed=2)
    index.build(range(100))
    path = tmp_path / "graph.npz"
    index.save(path)
    restored = HNSW.load(path, vectors)
    for query in vectors[100:110]:
        a, b = index.search(query), restored.search(query)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    # RNG state and adjacency must survive so continuation builds the same graph.
    index.build()
    restored.build()
    np.testing.assert_array_equal(index.levels, restored.levels)
    assert index.base == restored.base
    assert index.upper == restored.upper
    with pytest.raises(ValueError, match="supplied vectors"):
        HNSW.load(path, vectors[::-1])


def test_identical_vectors_and_empty_index(tmp_path):
    vectors = np.tile(np.array([[1, 0]], dtype=np.float32), (30, 1))
    index = HNSW(vectors, M=4, ef_construction=20)
    assert index.search([1, 0])[0].size == 0
    index.save(tmp_path / "empty.npz")
    assert HNSW.load(tmp_path / "empty.npz", vectors).size == 0
    index.build()
    ids, scores = index.search([1, 0], k=10, ef_search=30)
    assert len(set(ids)) == 10
    np.testing.assert_allclose(scores, 1)
    index.validate()


@pytest.mark.parametrize("query", [[0, 0], [np.nan, 1], [1, 2, 3]])
def test_rejects_invalid_queries(query):
    with pytest.raises(ValueError):
        HNSW(np.eye(2, dtype=np.float32)).build().search(query)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="normalized"):
        HNSW(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="unique"):
        HNSW(np.eye(2), ids=[10, 10])
    with pytest.raises(ValueError, match="ef_construction"):
        HNSW(np.eye(2), M=16, ef_construction=8)
    with pytest.raises(ValueError, match="ef_search"):
        HNSW(np.eye(2)).build().search([1, 0], k=10, ef_search=5)
