"""Does mean-centering improve retrieval quality? Raw vs centered cosine.

CLAP's audio space is strongly anisotropic (mean pairwise cosine ~0.87, top
principal component carries ~45% of variance). Mean-centering collapses that to
~0.04. This measures whether the geometry change buys real retrieval quality,
using genre agreement as a proxy: what fraction of a track's top-k neighbours
share its genre_top label.

Chance baseline is 17.2% (sum of squared genre shares) -- the corpus is skewed,
28% Rock and 25% Electronic, so a naive "always Rock" guess already scores well.
Only a score clearly above 17.2% means the space is discriminating.

Usage: PYTHONPATH=src python3 tests/centering_test.py [n_queries]
"""
import sqlite3
import sys

import numpy as np

DB = "store/timbre.db"
STORE = "store/vectors.npy"
K = 10


def load_tracks():
    """Track-level vectors (mean of each track's real windows) + genre labels."""
    store = np.load(STORE, mmap_mode="r")
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT row_start, n_windows, genre, title, artist FROM tracks "
        "WHERE status='done' AND n_windows > 0 AND genre IS NOT NULL"
    ).fetchall()
    conn.close()

    vecs = np.empty((len(rows), 512), dtype=np.float32)
    genres, meta = [], []
    for i, (rs, n, genre, title, artist) in enumerate(rows):
        vecs[i] = np.asarray(store[rs:rs + n]).mean(0)
        genres.append(genre)
        meta.append((title, artist, genre))
    return vecs, np.array(genres), meta


def normalize(V):
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def genre_precision(V, genres, queries, k=K):
    """Fraction of each query's top-k neighbours sharing its genre."""
    scores = []
    for qi in queries:
        sims = V @ V[qi]
        sims[qi] = -np.inf  # exclude self
        top = np.argpartition(-sims, k)[:k]
        scores.append((genres[top] == genres[qi]).mean())
    return float(np.mean(scores))


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"loading {STORE} ...")
    V, genres, meta = load_tracks()
    print(f"tracks: {len(V)}, genres: {len(set(genres))}")

    raw = normalize(V.copy())
    centered = normalize(V - V.mean(0))

    rng = np.random.default_rng(0)
    queries = rng.choice(len(V), n_queries, replace=False)

    # Chance baseline from the actual genre distribution
    _, counts = np.unique(genres, return_counts=True)
    chance = float(((counts / counts.sum()) ** 2).sum())

    p_raw = genre_precision(raw, genres, queries)
    p_cen = genre_precision(centered, genres, queries)

    def spread(M):
        sub = M[rng.choice(len(M), 1500, replace=False)]
        S = sub @ sub.T
        off = S[~np.eye(len(S), dtype=bool)]
        return off.mean(), off.std()

    m_raw, s_raw = spread(raw)
    m_cen, s_cen = spread(centered)

    print(f"\n{'':12s} {'genre@10':>10s} {'mean cos':>10s} {'std':>8s}")
    print(f"{'chance':12s} {chance:9.1%}")
    print(f"{'raw':12s} {p_raw:9.1%} {m_raw:10.3f} {s_raw:8.3f}")
    print(f"{'centered':12s} {p_cen:9.1%} {m_cen:10.3f} {s_cen:8.3f}")

    lift = p_cen - p_raw
    print(f"\ncentering changes genre@10 by {lift:+.1%} "
          f"({p_raw:.1%} -> {p_cen:.1%}, chance {chance:.1%})")
    if abs(lift) < 0.02:
        print("=> no meaningful difference; keep the simpler raw space")
    elif lift > 0:
        print("=> centering helps; index the centered space")
    else:
        print("=> centering hurts; keep raw")


if __name__ == "__main__":
    main()
