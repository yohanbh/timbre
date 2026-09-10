"""Compare a candidate CLAP checkpoint with the existing store on identical audio.

Run from the repo root:
    PYTHONPATH=src python3 tests/embedding_quality.py

This is a diagnostic, not an index-recall test. It reads the production store
without modifying it; optional --out saves the sample for listening/inspection.
Genre agreement is only a weak proxy for perceived similarity.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sqlite3

import numpy as np
import torch
from huggingface_hub import model_info
from transformers import ClapModel, ClapProcessor

from timbre.audio import decode
from timbre.embed import SR, WINDOW_SAMPLES, set_determinism


TEXTS = [
    "a heavy metal guitar solo",
    "a slow sad piano ballad",
    "a female opera singer",
    "fast electronic dance music",
]


def quality(vectors, genres):
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = vectors @ vectors.T
    off = sims[~np.eye(len(sims), dtype=bool)]
    np.fill_diagonal(sims, -np.inf)
    top = np.argsort(-sims, axis=1)[:, :10]
    return {
        "genre_at_10": float((genres[top] == genres[:, None]).mean()),
        "mean_pairwise_cosine": float(off.mean()),
        "mean_vector_norm": float(np.linalg.norm(vectors.mean(0))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="laion/larger_clap_general")
    ap.add_argument("--revision", help="defaults to the resolved current commit")
    ap.add_argument("--tracks", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--db", default="store/timbre.db")
    ap.add_argument("--store", default="store/vectors.npy")
    ap.add_argument("--out", help="optional diagnostic .npz (use a new path)")
    a = ap.parse_args()
    if a.out and os.path.exists(a.out):
        ap.error("--out already exists; choose a new path")

    with sqlite3.connect(f"file:{a.db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT track_id, path, row_start, n_windows, genre FROM tracks "
            "WHERE status='done' AND n_windows > 0 AND genre IS NOT NULL "
            "ORDER BY track_id"
        ).fetchall()
        source_meta = dict(conn.execute("SELECT key, value FROM meta"))
    if not 11 <= a.tracks <= len(rows):
        ap.error(f"--tracks must be between 11 and {len(rows)}")
    rng = np.random.default_rng(a.seed)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), a.tracks, replace=False))]
    offsets = [int(rng.integers(r[3])) for r in rows]
    row_ids = np.array([r[2] + offset for r, offset in zip(rows, offsets)])
    genres = np.array([r[4] for r in rows])
    baseline = np.array(np.load(a.store, mmap_mode="r")[row_ids])
    _, counts = np.unique(genres, return_counts=True)
    chance = float(np.sum(counts * (counts - 1)) / (len(rows) * (len(rows) - 1)))
    print("stored:", quality(baseline, genres), "chance:", chance, flush=True)

    revision = a.revision or model_info(a.model).sha
    print("loading", a.model, revision, flush=True)
    set_determinism()
    model, loading = ClapModel.from_pretrained(
        a.model, revision=revision, output_loading_info=True
    )
    if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
        raise RuntimeError(f"checkpoint did not load cleanly: {loading}")
    model = model.eval().cuda()
    processor = ClapProcessor.from_pretrained(a.model, revision=revision)
    tokens = processor(text=TEXTS, padding=True, return_tensors="pt")
    with torch.inference_mode():
        text_vectors = model.get_text_features(
            **{k: v.cuda() for k, v in tokens.items()}
        ).cpu().numpy()
    print("text cosine:\n", text_vectors @ text_vectors.T, flush=True)

    candidate = []
    for start in range(0, len(rows), 16):
        windows = []
        for row, offset in zip(rows[start:start + 16], offsets[start:start + 16]):
            audio = decode(row[1])
            window = audio[offset * SR:offset * SR + WINDOW_SAMPLES]
            assert window.size == WINDOW_SAMPLES
            windows.append(window)
        features = processor(audios=windows, sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            vectors = model.get_audio_features(
                **{k: v.cuda() for k, v in features.items()}
            ).cpu().numpy()
        candidate.append(vectors)
        print(f"  {min(start + 16, len(rows))}/{len(rows)}", flush=True)
    candidate = np.concatenate(candidate)
    report = {
        "model": a.model, "revision": revision, "source_meta": source_meta,
        "seed": a.seed, "tracks": len(rows), "chance_genre_at_10": chance,
        "stored": quality(baseline, genres), "candidate": quality(candidate, genres),
    }
    print(json.dumps(report, indent=2), flush=True)
    if a.out:
        np.savez(a.out, row_ids=row_ids, track_ids=np.array([r[0] for r in rows]),
                 genres=genres, stored=baseline, candidate=candidate,
                 texts=np.array(TEXTS), text_vectors=text_vectors,
                 report=json.dumps(report))


if __name__ == "__main__":
    main()
