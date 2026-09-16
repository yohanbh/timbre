"""Blind A/B: how far apart must two candidates be before you can hear it?

    # Gap-stratified (default): equal trials per cosine-gap band.
    USE_TF=0 PYTHONPATH=src python3 tests/blind_ab.py --trials 40

    # Fixed-rank, the original session-1 protocol.
    USE_TF=0 PYTHONPATH=src python3 tests/blind_ab.py --mode rank --near 1 --far 40

Plays the query passage, then two candidates in random order: one closer to the
query than the other. You pick which sounded closer. Nothing reveals which is
which until the end.

Gap mode samples pairs so each band of `near_score - far_score` gets comparable
coverage, because rank does not control gap: over 60 sampled queries the rank-1
to rank-40 gap ranged from 0.042 to 0.383. The reported output is accuracy per
band, which locates the similarity threshold below which ranking is inaudible.
A single overall accuracy number cannot show that.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import argparse
import json
import sqlite3
import subprocess
from math import comb

import numpy as np

from timbre.embed import HOP_SECONDS
from timbre.groundtruth import load_layout
from timbre.search import exact_track_topk

# Edges measured from the real score distribution over 60 sampled queries:
# pooled rank-pair gaps run p10=0.007, median=0.044, p90=0.115.
DEFAULT_BANDS = (0.0, 0.02, 0.05, 0.10, 1.0)
CANDIDATE_DEPTH = 60


def play(path, offset):
    subprocess.run(["ffplay", "-v", "error", "-autoexit", "-ss", str(offset),
                    "-t", "10", "-nodisp", path], check=False)


def binomial_p(correct, n):
    """One-sided probability of >= `correct` successes under a fair coin."""
    return sum(comb(n, i) for i in range(correct, n + 1)) / 2 ** n if n else 1.0


def band_of(gap, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= gap < edges[i + 1]:
            return i
    return len(edges) - 2


def draw_pair(rng, store, row_ids, owner, T, pool, mode, near, far, edges, wanted):
    """Return (query_track, window, near_tuple, far_tuple, gap) or None.

    In gap mode `wanted` names the band we still need; the pair is chosen from
    the candidate list to land in it. Rank does not determine gap, so the pair
    must be selected by score difference directly.
    """
    tid = int(rng.choice(pool))
    t = T[tid]
    w = int(rng.integers(0, t[2]))
    q = np.asarray(store[t[1] + w])
    depth = CANDIDATE_DEPTH if mode == "gap" else max(near, far)
    ids, hits, scores = exact_track_topk(
        store, q, row_ids, owner, k=depth, exclude_track=tid)
    if len(ids) < depth:
        return None

    def entry(label, i):
        cid = int(ids[i])
        return (label, cid, int(hits[i] - T[cid][1]), float(scores[i]))

    if mode == "rank":
        ni, fi = near - 1, far - 1
        return tid, w, entry("near", ni), entry("far", fi), float(scores[ni] - scores[fi])

    # Gap mode: the nearer candidate is always rank 1, so the gap is a pure
    # function of which farther candidate we pick.
    lo, hi = edges[wanted], edges[wanted + 1]
    options = [i for i in range(1, len(scores)) if lo <= scores[0] - scores[i] < hi]
    if not options:
        return None
    fi = int(rng.choice(options))
    return tid, w, entry("near", 0), entry("far", fi), float(scores[0] - scores[fi])


def summarize(log, edges, mode):
    n = len(log)
    correct = sum(1 for e in log if e["picked"] == "near")
    p = binomial_p(correct, n)
    print(f"\n{correct}/{n} correct ({correct/n*100:.0f}%). Chance is 50%.")
    print(f"One-sided binomial p = {p:.4f}"
          f"{'  (significant at 0.05)' if p < 0.05 else '  (not significant)'}")

    bands = []
    if mode == "gap":
        print(f"\n{'cosine gap':<16}{'correct':>9}{'acc':>7}{'p':>9}")
        for b in range(len(edges) - 1):
            rows = [e for e in log if e["band"] == b]
            if not rows:
                continue
            c = sum(1 for e in rows if e["picked"] == "near")
            bp = binomial_p(c, len(rows))
            label = (f"{edges[b]:.2f}-{edges[b+1]:.2f}" if edges[b + 1] < 1.0
                     else f"{edges[b]:.2f}+")
            print(f"{label:<16}{c:>4}/{len(rows):<4}{c/len(rows)*100:>6.0f}%{bp:>9.4f}")
            bands.append({"band": b, "low": edges[b], "high": edges[b + 1],
                          "correct": c, "trials": len(rows), "p_value": bp})
        audible = [x for x in bands if x["trials"] >= 5 and x["correct"] / x["trials"] >= 0.7]
        if audible:
            print(f"\nLowest band reaching 70% accuracy: "
                  f"{audible[0]['low']:.2f}-{audible[0]['high']:.2f}")
        else:
            print("\nNo band reached 70% accuracy with >= 5 trials.")

    replayed = [e for e in log if e["replays"]]
    first = [e for e in log if not e["replays"]]
    if replayed:
        rc = sum(1 for e in replayed if e["picked"] == "near")
        fc = sum(1 for e in first if e["picked"] == "near")
        print(f"\nReplayed {len(replayed)}/{n}: {rc}/{len(replayed)} correct"
              f"; first-listen {fc}/{len(first)} correct")
    return correct, p, bands, len(replayed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--mode", choices=["gap", "rank"], default="gap",
                    help="gap: stratify by cosine gap (default); rank: fixed ranks")
    ap.add_argument("--near", type=int, default=1, help="rank mode: close candidate")
    ap.add_argument("--far", type=int, default=40, help="rank mode: distant candidate")
    ap.add_argument("--bands", type=float, nargs="+", default=list(DEFAULT_BANDS),
                    help="gap mode: band edges, ascending")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--out", default="docs/listening_blind_ab.json")
    a = ap.parse_args()
    edges = sorted(a.bands)
    if a.trials < 1 or len(edges) < 2 or edges[0] < 0:
        ap.error("require trials >= 1 and at least two nonnegative band edges")
    if a.mode == "rank" and (a.near < 1 or a.far <= a.near):
        ap.error("require far > near >= 1")

    row_ids, owner = load_layout(a.db)
    store = np.load(a.store, mmap_mode="r")
    with sqlite3.connect(f"file:{a.db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT track_id,row_start,n_windows,title,artist,genre,path "
            "FROM tracks WHERE status='done' AND n_windows>0"
        ).fetchall()
    T = {r[0]: r for r in rows}
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(seed)
    pool = [r[0] for r in rows if r[2] >= 15]

    nbands = len(edges) - 1
    if a.mode == "gap":
        # Round-robin the bands so an interrupted session stays balanced.
        schedule = [b % nbands for b in range(a.trials)]
        rng.shuffle(schedule)
        print(f"Blind A/B, gap-stratified over {nbands} bands. Seed {seed}.")
        print("Bands: " + ", ".join(
            f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(nbands)))
    else:
        schedule = [0] * a.trials
        print(f"Blind A/B: rank {a.near} vs rank {a.far}. Seed {seed}.")
    print("Type 1 or 2 for whichever sounded closer to the query.")
    print("'r' replays the trial, 's' skips it, 'q' stops early.\n")

    log = []
    for trial, wanted in enumerate(schedule, 1):
        drawn = None
        for _ in range(40):          # bands near the tails need a few draws
            drawn = draw_pair(rng, store, row_ids, owner, T, pool,
                              a.mode, a.near, a.far, edges, wanted)
            if drawn:
                break
        if not drawn:
            print(f"--- trial {trial}: no pair found for this band, skipping")
            continue
        tid, w, near_e, far_e, gap = drawn
        pair = [near_e, far_e]
        if rng.random() < 0.5:
            pair.reverse()

        print(f"--- trial {trial}/{len(schedule)} --- query: track {tid} @ {w*HOP_SECONDS}s")
        replays = 0
        while True:
            print(f"  playing query...{'  (replay %d)' % replays if replays else ''}")
            play(T[tid][6], w * HOP_SECONDS)
            for n, (_, cid, coff, _) in enumerate(pair, 1):
                print(f"  playing {n}...")
                play(T[cid][6], coff * HOP_SECONDS)
            answer = input("  closer (1/2/r=replay/s=skip/q=quit)? ").strip().lower()
            if answer != "r":
                break
            replays += 1
        if answer == "q":
            break
        if answer not in ("1", "2"):
            continue
        log.append({"trial": trial, "query_track": tid, "query_window": w,
                    "picked": pair[int(answer) - 1][0], "replays": replays,
                    "band": band_of(gap, edges), "gap": gap,
                    "near_track": near_e[1], "far_track": far_e[1],
                    "near_score": near_e[3], "far_score": far_e[3]})

    if not log:
        print("\nNo scored trials.")
        return
    correct, p, bands, replayed = summarize(log, edges, a.mode)
    json.dump({"seed": seed, "mode": a.mode, "bands": edges,
               "near_rank": a.near, "far_rank": a.far,
               "correct": correct, "trials": len(log), "p_value": p,
               "replayed_trials": replayed, "band_results": bands, "log": log},
              open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
