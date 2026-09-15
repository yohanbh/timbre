"""Blind A/B: can you hear the difference between rank 1 and rank 40?

    USE_TF=0 PYTHONPATH=src python3 blind_ab.py --trials 20

Plays the query passage, then two candidates in random order: one from the
top of the ranking, one from deep in it. You pick which sounded closer.
Nothing reveals which is which until the end.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import argparse
import json
import sqlite3
import subprocess
import numpy as np

from timbre.embed import HOP_SECONDS
from timbre.groundtruth import load_layout
from timbre.search import exact_track_topk


def play(path, offset):
    subprocess.run(["ffplay", "-v", "error", "-autoexit", "-ss", str(offset),
                    "-t", "10", "-nodisp", path], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--near", type=int, default=1, help="rank of the close candidate")
    ap.add_argument("--far", type=int, default=40, help="rank of the distant candidate")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--out", default="docs/listening_blind_ab.json")
    a = ap.parse_args()

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

    log, correct = [], 0
    print(f"Blind A/B: rank {a.near} vs rank {a.far}. Seed {seed}.")
    print("After each pair, type 1 or 2 for whichever sounded closer to the query.")
    print("Type 's' to skip, 'q' to stop early.\n")

    for trial in range(1, a.trials + 1):
        tid = int(rng.choice(pool))
        t = T[tid]
        w = int(rng.integers(0, t[2]))
        q = np.asarray(store[t[1] + w])
        ids, hits, scores = exact_track_topk(
            store, q, row_ids, owner, k=max(a.near, a.far), exclude_track=tid)
        if len(ids) < a.far:
            continue
        ni, fi = a.near - 1, a.far - 1
        pair = [("near", int(ids[ni]), int(hits[ni] - T[int(ids[ni])][1]), float(scores[ni])),
                ("far", int(ids[fi]), int(hits[fi] - T[int(ids[fi])][1]), float(scores[fi]))]
        if rng.random() < 0.5:
            pair.reverse()

        print(f"--- trial {trial}/{a.trials} --- query: track {tid} @ {w*HOP_SECONDS}s")
        replays = 0
        while True:
            print(f"  playing query...{'  (replay %d)' % replays if replays else ''}")
            play(t[6], w * HOP_SECONDS)
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
        picked = pair[int(answer) - 1][0]
        correct += picked == "near"
        log.append({"trial": trial, "query_track": tid, "query_window": w,
                    "picked": picked, "pair": [p[0] for p in pair], "replays": replays,
                    "near_score": [p[3] for p in pair if p[0] == "near"][0],
                    "far_score": [p[3] for p in pair if p[0] == "far"][0]})

    n = len(log)
    if not n:
        print("\nNo scored trials.")
        return
    from math import comb
    p = sum(comb(n, i) for i in range(correct, n + 1)) / 2 ** n
    print(f"\n{correct}/{n} correct ({correct/n*100:.0f}%). Chance is 50%.")
    print(f"One-sided binomial p = {p:.4f}"
          f"{'  (significant at 0.05)' if p < 0.05 else '  (not significant)'}")
    replayed = [e for e in log if e["replays"]]
    if replayed:
        hard = sum(1 for e in replayed if e["picked"] == "near")
        easy = [e for e in log if not e["replays"]]
        easy_right = sum(1 for e in easy if e["picked"] == "near")
        print(f"Replayed {len(replayed)}/{n} trials: {hard}/{len(replayed)} correct"
              f"; first-listen trials {easy_right}/{len(easy)} correct")
    json.dump({"seed": seed, "near_rank": a.near, "far_rank": a.far,
               "correct": correct, "trials": n, "p_value": p,
               "replayed_trials": len(replayed), "log": log},
              open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
