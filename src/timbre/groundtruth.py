"""Phase 1: exact brute-force search and the evaluation harness.

This exists so Phase 2 can tell a working HNSW graph from a broken one. The
spec's own risk note is that a naive neighbour-selection heuristic yields a
graph that looks fine and searches badly, and this corpus has a weak geometric
signal, so eyeballing results is not a check.

Two ground-truth variants are cached per query, because one list cannot answer
both questions we will ask of it:

  unfiltered      true nearest neighbours. The honest target for index recall --
                  an exact index must reproduce exactly this.
  sibling-excluded  same-track windows removed. Measured on this corpus, 73% of
                  top-10 neighbours are windows of the *query's own track* and
                  39% of queries have an all-sibling top-10 (within-track
                  similarity is 0.979). Scoring retrieval quality or the Phase 2
                  aggregation rule against the unfiltered list would mostly be
                  measuring "can it find the same song again", which is not the
                  question.

Both are exact; they differ only in candidate eligibility.
"""
import sqlite3

import numpy as np

from .embed import DIM

# 50k x 512 float32 is ~100 MiB per chunk -- the full 510k x 512 pass would be
# ~1 GiB, which does not fit comfortably on a 7.4 GiB box alongside torch.
CHUNK_ROWS = 50_000

N_QUERIES = 1_000
TOP_K = 100
SEED = 20260909


def load_layout(db_path):
    """Return (row_ids, owner_track) for every populated row, ordered by track.

    The store is allocated at 21 windows per track but most tracks fill 20, so
    the raw array contains zero-filled holes (see verify.check_holes_and_norms).
    Every read path must go through this, never through a bare arange over the
    mmap, or those holes enter the search as fake unit-norm-zero vectors.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT track_id, row_start, n_windows FROM tracks "
        "WHERE status = 'done' ORDER BY track_id"
    ).fetchall()
    conn.close()

    row_ids = np.concatenate([np.arange(rs, rs + nw) for _, rs, nw in rows])
    owner = np.concatenate([np.full(nw, tid) for tid, _, nw in rows])
    return row_ids, owner


def owner_lookup(row_ids, owner):
    """Dense row_id -> track_id map. Holes map to -1 so a leaked hole is loud."""
    lut = np.full(row_ids.max() + 1, -1, dtype=np.int64)
    lut[row_ids] = owner
    return lut


def sample_queries(db_path, n=N_QUERIES, seed=SEED):
    """Genre-stratified query segments, reproducible from the seed alone.

    The corpus is 28% Rock / 25% Electronic, so a uniform sample is mostly those
    two and says little about the other 14 genres. Allocation is proportional
    with a floor of 1, so Easy Listening (21 tracks) is still represented.

    One window per sampled track, not several: sibling windows are 0.979 similar,
    so a second window of the same track is close to a duplicate query and would
    buy less than a different track for the same cost.
    """
    conn = sqlite3.connect(db_path)
    by_genre = {}
    for tid, rs, nw, genre in conn.execute(
        "SELECT track_id, row_start, n_windows, COALESCE(genre, '(none)') "
        "FROM tracks WHERE status = 'done' ORDER BY track_id"
    ):
        by_genre.setdefault(genre, []).append((tid, rs, nw))
    conn.close()

    total = sum(len(v) for v in by_genre.values())
    rng = np.random.default_rng(seed)

    # Deterministic order: sort genres by name, never by dict insertion.
    genres = sorted(by_genre)
    quota = {g: max(1, round(n * len(by_genre[g]) / total)) for g in genres}

    # Rounding + the floor will not sum to exactly n; trim or pad on the
    # largest genres so the total is exactly n and stays reproducible.
    while sum(quota.values()) > n:
        g = max(genres, key=lambda g: (quota[g], g))
        if quota[g] > 1:
            quota[g] -= 1
    while sum(quota.values()) < n:
        g = max(genres, key=lambda g: (len(by_genre[g]) - quota[g], g))
        quota[g] += 1

    q_rows, q_tracks, q_genres = [], [], []
    for g in genres:
        tracks = by_genre[g]
        pick = rng.choice(len(tracks), size=min(quota[g], len(tracks)), replace=False)
        for i in sorted(pick):
            tid, rs, nw = tracks[i]
            q_rows.append(rs + int(rng.integers(nw)))  # one window, uniform in track
            q_tracks.append(tid)
            q_genres.append(g)

    order = np.argsort(np.asarray(q_rows), kind="stable")
    return (np.asarray(q_rows)[order],
            np.asarray(q_tracks)[order],
            np.asarray(q_genres, dtype=object)[order])


def exact_topk(store, query_rows, row_ids, lut, query_tracks=None, k=TOP_K,
               exclude_siblings=False, chunk=CHUNK_ROWS):
    """Exact cosine top-k by chunked brute force.

    Vectors are L2-normalized at ingest (verify enforces it), so cosine is a
    plain dot product and no renormalization is needed here.

    The query itself is always excluded. With exclude_siblings, every window of
    the query's own track is excluded too.

    Returns (ids, sims), both (n_queries, k), sorted by descending similarity.
    """
    Q = np.asarray(store[query_rows], dtype=np.float32)
    nq = Q.shape[0]
    if Q.shape[1] != DIM:
        raise ValueError(f"expected dim {DIM}, got {Q.shape[1]}")

    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)

    for start in range(0, row_ids.size, chunk):
        sub = row_ids[start:start + chunk]
        B = np.asarray(store[sub], dtype=np.float32)
        S = Q @ B.T

        # Mask ineligible candidates to -inf rather than dropping columns, so
        # every query keeps the same chunk geometry.
        if exclude_siblings:
            S[lut[sub][None, :] == query_tracks[:, None]] = -np.inf
        else:
            S[sub[None, :] == query_rows[:, None]] = -np.inf

        kk = min(k, S.shape[1])
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        cand_s = np.concatenate([best_s, np.take_along_axis(S, part, axis=1)], axis=1)
        cand_i = np.concatenate([best_i, sub[part]], axis=1)

        # Tie-break by row id so the cached result does not depend on chunking.
        order = np.lexsort((cand_i, -cand_s), axis=1)[:, :k]
        best_s = np.take_along_axis(cand_s, order, axis=1)
        best_i = np.take_along_axis(cand_i, order, axis=1)

    return best_i, best_s


def recall_at_k(candidate_ids, truth_ids, k):
    """Mean fraction of the true top-k that a candidate index also returned.

    Set overlap, not rank correlation: an index that finds the right neighbours
    in a different order has not lost anything retrieval uses.
    """
    if k > truth_ids.shape[1]:
        raise ValueError(f"k={k} exceeds cached ground truth ({truth_ids.shape[1]})")
    hits = [
        np.intersect1d(candidate_ids[i, :k], truth_ids[i, :k]).size
        for i in range(truth_ids.shape[0])
    ]
    return float(np.mean(hits) / k)
