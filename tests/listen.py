"""Print a query track and its nearest neighbours, with ffplay commands.

Genre labels are a weak proxy for "sounds similar" -- this exists so a human can
check whether the retrieval is musically sensible. Prints paths; pass --play to
actually play 10 s of each through ffplay.

Usage:
    PYTHONPATH=src python3 tests/listen.py              # random query track
    PYTHONPATH=src python3 tests/listen.py --track 1234 # specific track id
    PYTHONPATH=src python3 tests/listen.py --text "sparse melancholy piano"
    PYTHONPATH=src python3 tests/listen.py --play       # play them
"""
import argparse
import sqlite3
import subprocess

import numpy as np

DB = "store/timbre.db"
STORE = "store/vectors.npy"


def load():
    store = np.load(STORE, mmap_mode="r")
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT track_id, row_start, n_windows, title, artist, genre, path "
        "FROM tracks WHERE status='done' AND n_windows > 0"
    ).fetchall()
    conn.close()
    V = np.empty((len(rows), 512), dtype=np.float32)
    for i, r in enumerate(rows):
        V[i] = np.asarray(store[r[1]:r[1] + r[2]]).mean(0)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, rows


def text_vector(query):
    """Embed a text query into the same space (CLAP is joint audio-text)."""
    import torch
    from transformers import ClapModel, ClapProcessor
    from timbre.embed import MODEL_ID, MODEL_REVISION
    model = ClapModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval().cuda()
    proc = ClapProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    t = proc(text=[query], return_tensors="pt", padding=True)
    with torch.inference_mode():
        e = model.get_text_features(**{k: v.cuda() for k, v in t.items()})
    e = e.cpu().numpy()[0]
    return e / np.linalg.norm(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, help="query by track id")
    ap.add_argument("--text", help="query by text phrase")
    ap.add_argument("--play", action="store_true", help="play 10 s of each via ffplay")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()

    V, rows = load()
    ids = [r[0] for r in rows]

    if a.text:
        q = text_vector(a.text)
        print(f'query: "{a.text}"\n')
    else:
        qi = ids.index(a.track) if a.track else np.random.default_rng().integers(len(V))
        q = V[qi]
        t = rows[qi]
        print(f"query: {t[3]!r} by {t[4]} [{t[5]}]  (track {t[0]})")
        if a.play:
            subprocess.run(["ffplay", "-v", "error", "-autoexit", "-t", "10", "-nodisp", t[6]])
        print()

    sims = V @ q
    if not a.text:
        sims[qi] = -np.inf
    for rank, i in enumerate(np.argsort(-sims)[:a.k], 1):
        t = rows[i]
        print(f"{rank}. {sims[i]:.3f}  {(t[3] or '?')[:36]:36s} {(t[4] or '?')[:18]:18s} [{t[5]}]")
        print(f"           {t[6]}")
        if a.play:
            subprocess.run(["ffplay", "-v", "error", "-autoexit", "-t", "10", "-nodisp", t[6]])


if __name__ == "__main__":
    main()
