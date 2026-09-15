"""Native search must retain the Python graph's behavior and cache identity."""
import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from timbre.hnsw import HNSW
from timbre.native_hnsw import NativeHNSW


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.mark.parametrize("dimension", [2, 13, 32, 512])
def test_search_matches_python_and_exact_at_full_beam(dimension):
    rng = np.random.default_rng(42)
    vectors = unit(rng.normal(size=(180, dimension)))
    index = HNSW(vectors, ids=rng.permutation(180) * 21, M=8, ef_construction=60)
    index.build(rng.permutation(180))
    native = NativeHNSW(index)
    for query in unit(rng.normal(size=(10, dimension))):
        for ef in (16, 64, 180):
            expected, expected_scores = index.search(query, ef_search=ef)
            actual, scores = native.search(query, ef_search=ef)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(scores, expected_scores, atol=1e-6)
            assert native.distance_evaluations == index.distance_evaluations
            if ef == 180:
                exact = index.ids[np.argsort(-(vectors @ query))[:10]]
                np.testing.assert_array_equal(actual, exact)


def test_empty_partial_snapshot_and_external_id_ties(tmp_path):
    vectors = np.tile(np.array([[1, 0]], dtype=np.float32), (30, 1))
    index = HNSW(vectors, ids=np.arange(30)[::-1] * 7, M=4, ef_construction=20)
    assert NativeHNSW(index).search([1, 0])[0].size == 0
    index.build([3, 7, 9])
    native = NativeHNSW(index)
    expected = sorted(index.ids[[3, 7, 9]])
    index.build()
    # The native snapshot must not include nodes inserted after conversion.
    np.testing.assert_array_equal(native.search([1, 0])[0], expected)
    index.save(tmp_path / "graph.npz")
    restored = NativeHNSW.load(tmp_path / "graph.npz", vectors)
    actual, scores = restored.search([1, 0], k=10, ef_search=30)
    np.testing.assert_array_equal(actual, index.search([1, 0], ef_search=30)[0])
    np.testing.assert_array_equal(scores, np.ones(10))
    with pytest.raises(ValueError, match="supplied vectors"):
        NativeHNSW.load(tmp_path / "graph.npz", np.roll(vectors, 1, axis=1))


def test_noncontiguous_vectors_queries_and_owner_lifetime():
    rng = np.random.default_rng(20)
    vectors = unit(rng.normal(size=(100, 24)))[:, ::-1]
    index = HNSW(vectors, M=8, ef_construction=60).build()
    native = NativeHNSW(index)
    queries = rng.normal(size=(20, 24)).astype(np.float32)[:, ::-1]
    expected = [index.search(q, ef_search=100)[0] for q in queries]
    del index, vectors
    gc.collect()
    # Traversal scratch state is local to each call while the GIL is released.
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(lambda q: native.search(q, ef_search=100)[0], queries))
    np.testing.assert_array_equal(actual, expected)


def test_direct_directory_matches_snapshot_and_retains_mmap_owners(tmp_path):
    rng = np.random.default_rng(31)
    vectors = unit(rng.normal(size=(120, 13)))
    builder = HNSW(vectors, ids=rng.permutation(120) * 9, M=8, ef_construction=60).build()
    expected = NativeHNSW(builder)
    path = tmp_path / "direct"
    expected.save_directory(path)
    loaded = NativeHNSW.load_directory(path, verify=True)
    assert isinstance(loaded.vectors, np.memmap)
    assert isinstance(loaded._layers[0][2], np.memmap)
    queries = unit(rng.normal(size=(8, 13)))
    wanted = [expected.search(query, ef_search=100) for query in queries]
    del expected, builder, vectors
    gc.collect()
    actual = [loaded.search(query, ef_search=100) for query in queries]
    for (wanted_ids, wanted_scores), (actual_ids, actual_scores) in zip(wanted, actual):
        np.testing.assert_array_equal(actual_ids, wanted_ids)
        np.testing.assert_array_equal(actual_scores, wanted_scores)


def test_direct_conversion_and_checksum_rejection(tmp_path):
    vectors = unit(np.random.default_rng(32).normal(size=(80, 8)))
    graph = HNSW(vectors, M=4, ef_construction=20).build()
    checkpoint, direct = tmp_path / "graph.npz", tmp_path / "direct"
    graph.save(checkpoint)
    loaded = NativeHNSW.convert(checkpoint, vectors, direct)
    np.testing.assert_array_equal(loaded.search(vectors[0])[0], graph.search(vectors[0])[0])
    ids = np.load(direct / "ids.npy", mmap_mode="r+")
    ids[0] += 1
    ids.flush()
    with pytest.raises(ValueError, match="checksum"):
        NativeHNSW.load_directory(direct, verify=True)


@pytest.mark.parametrize("query", [[0, 0], [np.nan, 1], [np.inf, 1], [1, 2, 3]])
def test_rejects_invalid_queries(query):
    with pytest.raises(ValueError):
        NativeHNSW(HNSW(np.eye(2, dtype=np.float32)).build()).search(query)


@pytest.mark.parametrize("k,ef", [(0, 10), (10, 9), (-1, 10)])
def test_rejects_invalid_beam(k, ef):
    with pytest.raises(ValueError, match="ef_search"):
        NativeHNSW(HNSW(np.eye(2, dtype=np.float32))).search([1, 0], k=k, ef_search=ef)


@pytest.mark.parametrize("nodes,offsets,links,entry", [
    ([0], [0, 2], [0], 0),  # offset beyond the links buffer
    ([0], [0, 1], [2], 0),  # link outside the vector matrix
    ([0], [0, 1], [1], 0),  # link to an uninserted node
    ([0], [0, 0], [], -1),  # invalid entry point
    ([0, 0], [0, 0, 0], [], 0),  # duplicate row mapping
])
def test_native_boundary_rejects_unsafe_graph_arrays(nodes, offsets, links, entry):
    from timbre._hnsw_native import SearchIndex

    layer = (np.array(nodes, dtype=np.int32), np.array(offsets, dtype=np.int64),
             np.array(links, dtype=np.int32))
    with pytest.raises(ValueError):
        SearchIndex(np.eye(2, dtype=np.float32), np.arange(2, dtype=np.int64), entry, [layer])
