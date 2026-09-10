"""Phase 1 verification: the four checks NEXT_STEPS.md specifies.

1. chunked brute force matches a naive loop
2. query sample is reproducible from the seed, and genre-stratified
3. cached neighbours reload identically
4. recall_at_k returns 1.0 against the ground truth itself
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from timbre.groundtruth import (SEED, exact_topk, load_layout, owner_lookup,
                                recall_at_k, sample_queries, load_groundtruth)

STORE_DIR = os.environ.get("TIMBRE_TEST_STORE_DIR", "store")
DB = os.path.join(STORE_DIR, "timbre.db")
STORE = os.path.join(STORE_DIR, "vectors.npy")
CACHE = os.path.join(STORE_DIR, "groundtruth.npz")


@pytest.fixture(scope="module")
def layout():
    row_ids, owner = load_layout(DB)
    return row_ids, owner, owner_lookup(row_ids, owner)


@pytest.fixture(scope="module")
def store():
    return np.load(STORE, mmap_mode="r")


# --- 1. chunked matmul == naive loop -----------------------------------------

def test_chunked_matches_naive_loop(layout, store):
    """The chunked top-k must equal a naive per-query scan over a subcorpus.

    Run on a slice of the corpus so the naive side is tractable, with a chunk
    size that forces several merges -- a single-chunk run would not exercise
    the merge path, which is where a chunked top-k actually goes wrong.
    """
    row_ids, _, lut = layout
    sub = row_ids[:20_000]
    q_rows = sub[np.random.default_rng(1).choice(sub.size, 100, replace=False)]
    q_tracks = lut[q_rows]

    ids, sims = exact_topk(store, q_rows, sub, lut, q_tracks, k=10, chunk=3_000)

    B = np.asarray(store[sub], dtype=np.float32)
    for i, qr in enumerate(q_rows):
        s = np.asarray(store[qr], dtype=np.float32) @ B.T
        s[sub == qr] = -np.inf
        naive = np.lexsort((sub, -s))[:10]
        assert np.array_equal(ids[i], sub[naive]), f"ids differ for query {qr}"
        np.testing.assert_allclose(sims[i], s[naive], rtol=0, atol=1e-6)


def test_chunk_size_does_not_change_result(layout, store):
    """Neighbour ids must not depend on chunk width. Sims may, in the last bits.

    Measured on this box: the same dot products computed at chunk=1500 vs
    chunk=20000 differ on ~1.3% of elements by up to 7.7e-07, in the raw BLAS
    output before any top-k logic. Chunk width changes the GEMM tile
    decomposition -- the same effect embed.py documents for batch composition.
    Ids are what recall_at_k consumes, so ids are the contract; asserting
    bit-identical sims across chunk widths would assert something the hardware
    does not promise.
    """
    row_ids, _, lut = layout
    sub = row_ids[:20_000]
    q_rows = sub[np.random.default_rng(2).choice(sub.size, 40, replace=False)]
    q_tracks = lut[q_rows]

    a = exact_topk(store, q_rows, sub, lut, q_tracks, k=20, chunk=1_500)
    b = exact_topk(store, q_rows, sub, lut, q_tracks, k=20, chunk=20_000)
    assert np.array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-5)


def test_self_is_excluded(layout, store):
    """A query must never retrieve itself; sim 1.0 at rank 0 would hide bugs."""
    row_ids, _, lut = layout
    sub = row_ids[:20_000]
    q_rows = sub[np.random.default_rng(3).choice(sub.size, 50, replace=False)]
    ids, _ = exact_topk(store, q_rows, sub, lut, lut[q_rows], k=10, chunk=4_000)
    assert not np.any(ids == q_rows[:, None])


def test_sibling_exclusion_removes_own_track(layout, store):
    row_ids, _, lut = layout
    sub = row_ids[:20_000]
    q_rows = sub[np.random.default_rng(4).choice(sub.size, 50, replace=False)]
    q_tracks = lut[q_rows]
    ids, _ = exact_topk(store, q_rows, sub, lut, q_tracks, k=10,
                        exclude_siblings=True, chunk=4_000)
    assert not np.any(lut[ids] == q_tracks[:, None])


# --- 2. reproducible, stratified query sample --------------------------------

def test_query_sample_reproducible():
    a = sample_queries(DB, 200, SEED)
    b = sample_queries(DB, 200, SEED)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(sample_queries(DB, 200, SEED + 1)[0], a[0])


def test_query_sample_is_stratified():
    """Every genre represented, and no genre swamping its corpus share."""
    _, tracks, genres = sample_queries(DB, 1000, SEED)
    assert len(tracks) == 1000
    assert len(np.unique(tracks)) == 1000, "one window per track"

    counts = {g: int((genres == g).sum()) for g in np.unique(genres)}
    assert len(counts) == 16, f"expected all 16 genres, got {len(counts)}"
    assert min(counts.values()) >= 1
    # Rock is 28% of the corpus; proportional allocation must stay near that.
    assert counts["Rock"] < 340


def test_query_rows_are_populated(layout):
    """Sampled rows must be real vectors, never allocation holes."""
    _, _, lut = layout
    q_rows, q_tracks, _ = sample_queries(DB, 300, SEED)
    assert np.all(lut[q_rows] != -1), "sampled a hole"
    assert np.array_equal(lut[q_rows], q_tracks)


# --- 3. cache round-trip ------------------------------------------------------

def _rebuild(out, env_extra=None):
    """Rebuild the cache in a subprocess with a scrubbed environment.

    BLAS thread vars are stripped rather than inherited: build_groundtruth pins
    them itself, and letting the caller's shell leak in would make this test
    pass or fail depending on how pytest was invoked.
    """
    import os
    env = {k: v for k, v in os.environ.items()
           if not k.endswith(("_NUM_THREADS",))}
    env["PYTHONPATH"] = "src"
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "timbre.build_groundtruth", DB, STORE, str(out)],
        capture_output=True, text=True, env=env,
    )


def test_cache_reloads_identically(tmp_path):
    """Rebuild into a temp file and confirm it matches the committed cache."""
    import os
    if not os.path.exists(CACHE):
        pytest.skip("ground truth not built yet")

    out = tmp_path / "gt.npz"
    r = _rebuild(out)
    assert r.returncode == 0, r.stderr

    a = load_groundtruth(CACHE, DB, STORE)
    b = load_groundtruth(out, DB, STORE)
    for key in ("query_rows", "gt_ids", "gt_ids_nosib"):
        assert np.array_equal(a[key], b[key]), f"{key} not reproducible"
    for key in ("gt_sims", "gt_sims_nosib"):
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0)


def test_build_pins_threads_against_ambient_env(tmp_path):
    """A hostile ambient OMP_NUM_THREADS must not change the cached result.

    Measured: at 1, 2 and 8 BLAS threads the top-100 differs from the 4-thread
    build in *ids*, not merely sims -- neighbours sit at 0.979 similarity, so a
    changed reduction order reorders near-ties. build_groundtruth therefore pins
    the count before numpy is imported; this test is what keeps that honest.
    """
    import os
    if not os.path.exists(CACHE):
        pytest.skip("ground truth not built yet")

    out = tmp_path / "gt_hostile.npz"
    r = _rebuild(out, {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1"})
    assert r.returncode == 0, r.stderr

    a, b = np.load(CACHE), np.load(out)
    assert int(b["blas_threads"]) == 4, "build did not pin its thread count"
    assert np.array_equal(a["gt_ids"], b["gt_ids"]), \
        "ambient thread env leaked into the cached ground truth"
    np.testing.assert_allclose(a["gt_sims"], b["gt_sims"], rtol=0, atol=0)


# --- 4. recall_at_k -----------------------------------------------------------

def test_recall_against_itself_is_one():
    truth = np.arange(500).reshape(50, 10)
    assert recall_at_k(truth, truth, 10) == 1.0
    assert recall_at_k(truth, truth, 5) == 1.0


def test_recall_is_order_insensitive():
    truth = np.arange(500).reshape(50, 10)
    assert recall_at_k(truth[:, ::-1], truth, 10) == 1.0


def test_recall_partial_and_zero():
    truth = np.arange(100).reshape(10, 10)
    half = truth.copy()
    half[:, 5:] = -np.arange(1, 51).reshape(10, 5)  # 5 of 10 wrong
    assert recall_at_k(half, truth, 10) == pytest.approx(0.5)
    assert recall_at_k(-truth - 1, truth, 10) == 0.0


def test_recall_rejects_k_beyond_cache():
    truth = np.arange(100).reshape(10, 10)
    with pytest.raises(ValueError):
        recall_at_k(truth, truth, 11)
