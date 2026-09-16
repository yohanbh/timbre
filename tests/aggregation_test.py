"""How should a track's 20 windows be reduced to one relevance score?

    USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 tests/aggregation_test.py 2000

The store keeps 20-21 windows per track. Ranking whole tracks requires collapsing
each candidate's windows into one score, and the choice was never measured --
`tests/centering_test.py` pools each track to its mean vector, while
`timbre.search.exact_track_topk` takes the best matching segment.

That mattered less when a track's windows were nearly one vector. On the
replacement checkpoint they are not: mean within-track pairwise cosine is 0.917
and first-versus-last is 0.821, and early-versus-late queries share a median of
only 2/10 neighbours (see LISTENING.md). So the rules can genuinely disagree.

Rules compared, all on identical query windows and candidates:

  max    best matching window, the current search behaviour
  mean   average over the candidate's real windows
  topk   mean of the candidate's best `--top` windows, between max and mean
  count  how many of the candidate's windows beat a similarity threshold

Scored by genre agreement among the top-10 distinct tracks, the same proxy
`centering_test.py` uses, against the same 17.2% chance baseline. Genre agreement
is a weak stand-in for musical similarity and cannot settle the question on its
own; LISTENING.md carries the human judgments.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import argparse
import json
import sqlite3
import time

import numpy as np

from timbre.groundtruth import CHUNK_ROWS, load_layout

RULES = ("max", "mean", "topk", "count")


def track_scores(sims, owner, starts, counts, rule, top, threshold):
    """Reduce per-window similarities to one score per candidate track."""
    n = len(starts)
    if rule == "max":
        out = np.full(n, -np.inf, dtype=np.float32)
        np.maximum.at(out, owner, sims)
        return out
    if rule == "count":
        hit = (sims >= threshold).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        np.add.at(out, owner, hit)
        # Ties on whole counts are broken by the best window, so a track with
        # one strong window outranks one with a single marginal window.
        best = np.full(n, -np.inf, dtype=np.float32)
        np.maximum.at(best, owner, sims)
        return out + np.clip(best, 0, None) * 1e-3
    if rule == "mean":
        total = np.zeros(n, dtype=np.float64)
        np.add.at(total, owner, sims.astype(np.float64))
        return (total / np.maximum(counts, 1)).astype(np.float32)
    # topk: mean of each track's best `top` windows. Tracks hold 20-21 windows,
    # so pad every block to the same width and sort once rather than looping
    # over 25,000 Python slices.
    width = int(counts.max())
    padded = np.full((n, width), -np.inf, dtype=np.float32)
    col = np.arange(len(sims)) - starts[owner]
    padded[owner, col] = sims
    padded.sort(axis=1)
    taken = padded[:, -top:] if top < width else padded
    valid = np.isfinite(taken)
    kept = np.maximum(valid.sum(1), 1)
    return (np.where(valid, taken, 0).sum(1) / kept).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("queries", type=int, nargs="?", default=500)
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--top", type=int, default=3, help="windows averaged by topk")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="similarity a window must beat for the count rule")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out", default="docs/aggregation_results.json")
    a = ap.parse_args()
    if a.queries < 1 or a.k < 1 or a.top < 1:
        ap.error("queries, k and top must be positive")

    row_ids, owner_track = load_layout(a.db)
    store = np.load(a.store, mmap_mode="r")
    with sqlite3.connect(f"file:{a.db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT track_id,row_start,n_windows,genre FROM tracks "
            "WHERE status='done' AND n_windows>0 AND genre IS NOT NULL "
            "ORDER BY track_id"
        ).fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    genres = np.array([r[3] for r in rows])
    index_of = {int(t): i for i, t in enumerate(ids)}

    # Restrict candidates to genre-labelled tracks, then lay their rows out
    # contiguously so each track owns one slice of the similarity vector.
    keep = np.array([int(t) in index_of for t in owner_track])
    rows_kept, owner_kept = row_ids[keep], owner_track[keep]
    order = np.argsort(owner_kept, kind="stable")
    rows_kept, owner_kept = rows_kept[order], owner_kept[order]
    owner_idx = np.array([index_of[int(t)] for t in owner_kept], dtype=np.int64)
    counts = np.bincount(owner_idx, minlength=len(ids)).astype(np.int64)
    starts = np.r_[0, np.cumsum(counts)[:-1]]

    shares = np.array([np.mean(genres == g) for g in set(genres)])
    chance = float((shares ** 2).sum())

    rng = np.random.default_rng(a.seed)
    picks = rng.choice(len(ids), size=min(a.queries, len(ids)), replace=False)
    print(f"{len(ids):,} genre-labelled tracks, {len(rows_kept):,} windows; "
          f"{len(picks)} queries, k={a.k}, chance={chance*100:.1f}%")

    agree = {r: [] for r in RULES}
    timing = {r: 0.0 for r in RULES}
    for n, qi in enumerate(picks, 1):
        # One mid-track window per query, matching how listening queries work.
        qrow = int(rows_kept[starts[qi] + counts[qi] // 2])
        q = np.asarray(store[qrow], dtype=np.float32)
        q = q / np.linalg.norm(q)
        sims = np.empty(len(rows_kept), dtype=np.float32)
        for s in range(0, len(rows_kept), CHUNK_ROWS):
            sub = rows_kept[s:s + CHUNK_ROWS]
            sims[s:s + len(sub)] = np.asarray(store[sub]) @ q
        for rule in RULES:
            started = time.perf_counter()
            scores = track_scores(sims, owner_idx, starts, counts, rule, a.top, a.threshold)
            timing[rule] += time.perf_counter() - started
            scores[qi] = -np.inf                      # exclude the query's own track
            top = np.argpartition(-scores, a.k)[:a.k]
            agree[rule].append(float(np.mean(genres[top] == genres[qi])))
        if n % 100 == 0:
            print(f"  {n}/{len(picks)}", flush=True)

    print(f"\n{'rule':<8}{'genre@' + str(a.k):>10}{'lift':>8}{'ms/query':>10}")
    report = {"queries": len(picks), "k": a.k, "top": a.top,
              "threshold": a.threshold, "chance": chance, "seed": a.seed,
              "tracks": len(ids), "windows": len(rows_kept), "rules": {}}
    for rule in RULES:
        mean = float(np.mean(agree[rule]))
        sem = float(np.std(agree[rule], ddof=1) / np.sqrt(len(agree[rule])))
        ms = timing[rule] / len(picks) * 1000
        label = f"{rule}@{a.top}" if rule == "topk" else (
            f"{rule}>{a.threshold}" if rule == "count" else rule)
        print(f"{label:<8}{mean*100:>9.1f}%{mean/chance:>7.1f}x{ms:>10.3f}")
        report["rules"][rule] = {"genre_at_k": mean, "sem": sem,
                                 "lift": mean / chance, "ms_per_query": ms}

    best = max(RULES, key=lambda r: report["rules"][r]["genre_at_k"])
    base = report["rules"]["max"]
    diff = report["rules"][best]["genre_at_k"] - base["genre_at_k"]
    sem = (report["rules"][best]["sem"] ** 2 + base["sem"] ** 2) ** 0.5
    report["best"] = best
    report["best_minus_max"] = {"difference": diff, "sem": sem}
    print(f"\nbest: {best}. Versus max (current search behaviour): "
          f"{diff*100:+.1f} points, SEM {sem*100:.1f}.")
    if abs(diff) < 2 * sem:
        print("That is within two standard errors: not a distinguishable difference.")
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
