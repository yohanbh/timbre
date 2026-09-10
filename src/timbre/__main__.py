"""Entrypoint that pins thread pools before numpy/torch are imported.

BLAS and OpenMP size their thread pools at import time, so these must be set
before `timbre.ingest` (and therefore numpy/torch) is imported at all. Setting
them later -- in run(), or inside a forked worker -- has no effect: each worker
then starts ~17 threads, and 6 workers put ~100 runnable threads on 16 cores,
which collapsed throughput to 0.83 tracks/s.

Run as: python -m timbre <args>
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"
os.environ.setdefault("USE_TF", "0")

import argparse  # noqa: E402

from .ingest import run  # noqa: E402

ap = argparse.ArgumentParser(prog="timbre")
ap.add_argument("--audio-root", default="data/fma_medium")
ap.add_argument("--db", default="store/timbre.db")
ap.add_argument("--store", default="store/vectors.npy")
ap.add_argument("--limit", type=int)
ap.add_argument("--csv", default="data/fma_metadata/tracks.csv")
a = ap.parse_args()
run(a.audio_root, a.db, a.store, a.limit, a.csv)
