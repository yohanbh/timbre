"""Construction parity, exact recall, and deterministic checkpoint continuation."""
import json

import numpy as np
import pytest

from timbre.hnsw import HNSW
from timbre.native_hnsw import NativeHNSW, NativeHNSWBuilder


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.mark.parametrize("dimension", [13, 512])
def test_small_build_matches_python_graph_and_exact_search(tmp_path, dimension):
    rng = np.random.default_rng(49)
    vectors = unit(rng.normal(size=(180, dimension)))
    ids, order = rng.permutation(180) * 21, rng.permutation(180)
    python = HNSW(vectors, ids, M=8, ef_construction=60, seed=4).build(order)
    native = NativeHNSWBuilder(vectors, ids, M=8, ef_construction=60, seed=4).build(order)
    native.validate()
    native.save(tmp_path / "native.npz")
    restored = HNSW.load(tmp_path / "native.npz", vectors)
    np.testing.assert_array_equal(python.levels, native.levels)
    assert python.base == restored.base
    assert python.upper == restored.upper
    snapshot = native.freeze()
    for query in unit(rng.normal(size=(8, dimension))):
        found, scores = snapshot.search(query, ef_search=180)
        rows = np.argsort(-(vectors @ query))[:10]
        expected = ids[rows]
        np.testing.assert_array_equal(found, expected)
        np.testing.assert_allclose(scores, vectors[rows] @ query, atol=1e-6)


def test_diversification_keeps_route_in_a_different_direction(tmp_path):
    angles = np.deg2rad([5, 6, -20, 0])
    vectors = unit(np.column_stack([np.cos(angles), np.sin(angles)]))
    native = NativeHNSWBuilder(vectors, M=2, ef_construction=4, seed=7).build()
    native.save(tmp_path / "graph.npz")
    restored = HNSW.load(tmp_path / "graph.npz", vectors)
    # For the last node, 5 and 6 degrees are redundant; preserve the -20 route.
    assert restored.base[3] == [0, 2]


def test_resumed_batches_preserve_graph_rng_and_search(tmp_path):
    rng = np.random.default_rng(6)
    vectors = unit(rng.normal(size=(1107, 16)))
    order = rng.permutation(len(vectors))
    complete = NativeHNSWBuilder(vectors, M=8, ef_construction=60, seed=8).build(order)
    partial = NativeHNSWBuilder(vectors, M=8, ef_construction=60, seed=8)
    partial.build(order[:137])
    snapshot = partial.freeze()
    path = tmp_path / "checkpoint.npz"
    partial.save(path)
    resumed = NativeHNSWBuilder.load(path, vectors)
    resumed.build(order[137:713])
    resumed.save(path)
    resumed = NativeHNSWBuilder.load(path, vectors)
    resumed.build(order[713:])
    assert snapshot.size == 137
    assert set(snapshot.search(vectors[0], ef_search=137)[0]) <= set(order[:137])
    resumed_arrays = resumed._builder.export_graph()
    for key, array in complete._builder.export_graph().items():
        np.testing.assert_array_equal(array, resumed_arrays[key])
    assert complete.rng.bit_generator.state == resumed.rng.bit_generator.state
    resumed.save(path)
    loaded = NativeHNSW.load(path, vectors)
    complete_snapshot = complete.freeze()
    for query in vectors[:5]:
        expected, actual = complete_snapshot.search(query), loaded.search(query)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
    with pytest.raises(ValueError, match="supplied vectors"):
        NativeHNSWBuilder.load(path, vectors[::-1])


def test_interrupted_progress_resumes_from_complete_checkpoint(tmp_path):
    vectors = unit(np.random.default_rng(2).normal(size=(2030, 8)))
    index = NativeHNSWBuilder(vectors, M=4, ef_construction=20, seed=3)
    path = tmp_path / "checkpoint.npz"

    def stop(n):
        if n == 1000:
            index.save(path)
        else:
            assert n == 2000
            raise InterruptedError  # Discard 1,000 insertions since the last save.

    with pytest.raises(InterruptedError):
        index.build(progress=stop)
    resumed = NativeHNSWBuilder.load(path, vectors)
    assert resumed.size == 1000
    resumed.build()
    complete = NativeHNSWBuilder(vectors, M=4, ef_construction=20, seed=3).build()
    expected = complete._builder.export_graph()
    for key, array in resumed._builder.export_graph().items():
        np.testing.assert_array_equal(array, expected[key])


def test_clustered_held_out_recall():
    rng = np.random.default_rng(9)
    centers = unit(rng.normal(size=(20, 32)))
    vectors = unit(centers[rng.integers(20, size=1000)] + 0.1 * rng.normal(size=(1000, 32)))
    queries = unit(centers[rng.integers(20, size=60)] + 0.1 * rng.normal(size=(60, 32)))
    native = NativeHNSWBuilder(vectors, M=12, ef_construction=80, seed=3).build(rng.permutation(1000))
    index = native.freeze()
    truth = np.argsort(-(queries @ vectors.T), axis=1)[:, :10]
    hits = [len(set(index.search(q, ef_search=80)[0]) & set(t)) for q, t in zip(queries, truth)]
    assert np.mean(hits) / 10 >= 0.95


def test_empty_singleton_identical_vectors_and_backend_identity(tmp_path):
    vectors = np.tile(np.array([[1, 0]], dtype=np.float32), (30, 1))
    index = NativeHNSWBuilder(vectors, ids=np.arange(30)[::-1], M=4, ef_construction=20)
    path = tmp_path / "empty.npz"
    index.save(path)
    index = NativeHNSWBuilder.load(path, vectors)
    assert index.freeze().search([1, 0])[0].size == 0
    index.insert(8)
    assert index.freeze().search([1, 0])[0].tolist() == [21]
    index.build()
    ids, scores = index.freeze().search([1, 0], ef_search=30)
    reference = HNSW(vectors, ids=np.arange(30)[::-1], M=4, ef_construction=20)
    reference.insert(8)
    reference.build()
    np.testing.assert_array_equal(ids, reference.search([1, 0], ef_search=30)[0])
    assert len(set(ids)) == 10
    np.testing.assert_array_equal(scores, np.ones(10))
    HNSW(vectors).save(path)
    with pytest.raises(ValueError, match="backend"):
        NativeHNSWBuilder.load(path, vectors)


@pytest.mark.parametrize("order", [[0, 0], [-1], [3], [[0, 1]]])
def test_invalid_order_does_not_advance_rng_or_mutate_graph(order):
    index = NativeHNSWBuilder(np.eye(3, dtype=np.float32))
    before = json.dumps(index.rng.bit_generator.state)
    with pytest.raises(ValueError, match="uninserted"):
        index.build(order)
    assert index.size == 0
    assert json.dumps(index.rng.bit_generator.state) == before
    index.insert(0)
    with pytest.raises(ValueError, match="uninserted"):
        index.insert(0)


def test_native_boundary_rejects_unsafe_insertion_and_restore():
    from timbre._hnsw_native import Builder

    builder = Builder(np.eye(3, dtype=np.float32), 2, 4)
    for order, levels in [([0, 0], [0, 0]), ([0, 4], [0, 0]), ([0], [-1]), ([0], [])]:
        with pytest.raises(ValueError):
            builder.build(np.array(order, dtype=np.int32), np.array(levels, dtype=np.int16))
        assert builder.size == 0
    with pytest.raises(ValueError, match="link"):
        builder.restore([[[2]], [], []], 0)
    assert builder.size == 0
