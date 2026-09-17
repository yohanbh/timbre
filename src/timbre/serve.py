"""FastAPI demo over the large index: one page, one input, five results.

    PYTHONPATH=src python3 -m timbre.serve --graph store/hnsw_locality_large \
      --db store/large/timbre.db

Phase 4 of TIMBRE_SPEC.md, and the only phase the spec calls optional. The
guardrails are binding: one page, five results, no third-party ANN library
anywhere in the serving path.

**No audio is served.** A query can return any of 105,884 tracks and that audio
is 99 GB; shipping a subset only works if the index is restricted to the same
subset, which would throw away the corpus the rest of the project measured. So a
result is a metadata card plus a YouTube *search* link, and the page says so.

The index is the hand-written graph from `native_hnsw`, memory-mapped at startup.
Nothing here computes distances; it maps rows back to tracks and gets out of the
way.
"""
import os

# BLAS and OpenMP size their pools when numpy/torch are imported, so the limits
# have to be set before those imports, not after. Getting this backwards cost a
# 6x throughput collapse in Phase 0; see the README.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("USE_TF", "0")

import argparse
import sqlite3
import subprocess
import tempfile
import threading
import urllib.parse
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .embed import HOP_SECONDS, MODEL_ID, MODEL_REVISION
from .groundtruth import load_layout, owner_lookup
from .manifest import check_versions
from .native_hnsw import NativeHNSW

RESULTS = 5
EF_SEARCH = 64
# Windows overlap 90%, so one track occupies many consecutive top rows and the
# raw hit list must be over-fetched before deduping. 50 is measured: it yields
# far more than 5 distinct tracks, and it keeps k <= ef_search=64 so the beam
# stays at the benchmarked width. Raising k past ef forces a wider beam --
# k=100 needs ef>=100 and measured 21 ms against 0.336 ms here, a 60x cost.
OVERFETCH = 50
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PAGE = Path(__file__).parent / "static" / "index.html"


class Demo:
    """Everything loaded once at startup and shared by every request."""

    def __init__(self, graph, db, device="cpu"):
        self.index = NativeHNSW.load_directory(graph)
        self.lock = threading.Lock()          # search mutates distance_evaluations
        row_ids, owner = load_layout(db)
        self.lut = owner_lookup(row_ids, owner)
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            self.tracks = {
                r[0]: r for r in conn.execute(
                    "SELECT track_id,row_start,n_windows,title,artist,genre,duration "
                    "FROM tracks WHERE status='done' AND n_windows>0"
                )
            }
            # The store was built on GPU torch; a CPU wheel can never report the
            # recorded +cu130 tag, so this is expected to fail here. It guards
            # bit-identical ingest, which a read-only query path cannot affect,
            # so record the drift and carry on rather than weakening the guard.
            try:
                check_versions(conn)
                self.version_drift = None
            except RuntimeError as error:
                self.version_drift = str(error)
        self.embedder, self.model_error = None, None
        try:
            from .embed import Embedder

            self.embedder = Embedder(device=device)
            self.embedder.check_text_separation()
        except Exception as error:                      # noqa: BLE001 - reported, not raised
            self.model_error = f"{type(error).__name__}: {error}"

    def warm(self, n=20):
        """Pay the cold costs at startup instead of on the first visitor.

        Two separate warm-ups. The graph needs its pages touched -- a cold page
        cache costs ~171x at p99 on this index. The model needs a real forward
        pass, because torch initializes lazily and the first text embed is
        several seconds even after the weights are loaded.
        """
        rng = np.random.default_rng(0)
        dim = self.index.vectors.shape[1]
        for _ in range(n):
            self.search(rng.standard_normal(dim).astype(np.float32))
        if self.embedder is not None:
            for _ in range(3):
                self.embedder.embed_text(["warming the text tower"])

    def search(self, query, k=RESULTS):
        """Rank distinct tracks by their best matching window.

        `NativeHNSW.search` returns absolute store rows with no per-track dedupe,
        so over-fetch and keep each track's best row. Ordering matches
        `search.exact_track_topk`: score descending, ties to the smaller row.
        """
        want = OVERFETCH
        for _ in range(3):
            with self.lock:
                rows, scores = self.index.search(
                    query, k=min(want, self.index.size),
                    ef_search=max(EF_SEARCH, want),   # the API requires ef >= k
                )
            rows = np.asarray(rows, dtype=np.int64)
            scores = np.asarray(scores, dtype=np.float32)
            keep = rows < len(self.lut)
            rows, scores = rows[keep], scores[keep]
            hits, seen = [], set()
            for i in np.lexsort((rows, -scores)):
                track = int(self.lut[rows[i]])
                if track < 0 or track in seen or track not in self.tracks:
                    continue
                seen.add(track)
                hits.append((track, int(rows[i]), float(scores[i])))
                if len(hits) == k:
                    return hits
            if want >= self.index.size:
                break
            want *= 4                        # too few distinct tracks; widens the beam
        return hits

    def cards(self, hits):
        out = []
        for rank, (track, row, score) in enumerate(hits, 1):
            t = self.tracks[track]
            offset = (row - t[1]) * HOP_SECONDS
            title, artist = t[3] or "Untitled", t[4] or "Unknown artist"
            out.append({
                "rank": rank, "track_id": track, "score": round(score, 4),
                "title": title, "artist": artist, "genre": t[5],
                "offset_s": offset, "passage": f"{_clock(offset)}-{_clock(offset + 10)}",
                "youtube": "https://www.youtube.com/results?search_query="
                           + urllib.parse.quote_plus(f"{artist} {title}"),
            })
        return out


def _clock(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _upload_query(demo, raw):
    """Mean of an uploaded clip's windows, embedded the way ingest does."""
    from .audio import DecodeError, decode, windows
    from .embed import assert_window_len

    with tempfile.NamedTemporaryFile(suffix=".upload") as handle:
        handle.write(raw)
        handle.flush()
        try:
            audio = decode(handle.name)
        except DecodeError as error:
            return None, f"could not decode that file: {error}"
    wins = windows(audio)
    if not wins:
        return None, "clip is shorter than the 10 second window"
    for w in wins:
        assert_window_len(w)
    return demo.embedder.embed(wins).mean(0), None


def create_app(demo):
    app = FastAPI(title="Timbre")

    @app.get("/", response_class=HTMLResponse)
    def page():
        html = PAGE.read_text()
        banner = ""
        if demo.model_error:
            banner = ("Text and upload queries are unavailable: the embedding "
                      f"model did not load ({demo.model_error}).")
        elif demo.version_drift:
            banner = ("This server embeds queries in a different environment than "
                      "the one that built the index, so results may drift slightly.")
        return html.replace("<!--BANNER-->", banner)

    @app.get("/health")
    def health():
        return {
            "nodes": int(demo.index.size),
            "tracks": len(demo.tracks),
            "model": MODEL_ID, "model_revision": MODEL_REVISION,
            "model_loaded": demo.embedder is not None,
            "model_error": demo.model_error,
            "version_drift": demo.version_drift,
            "ef_search": EF_SEARCH, "results": RESULTS,
        }

    @app.post("/search")
    async def search(text: str = Form(None), clip: UploadFile = File(None)):
        if demo.embedder is None:
            return JSONResponse({"error": demo.model_error or "model unavailable"},
                                status_code=503)
        if clip is not None and clip.filename:
            raw = await clip.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                return JSONResponse({"error": "clip too large (20 MB limit)"},
                                    status_code=413)
            query, error = _upload_query(demo, raw)
            if error:
                return JSONResponse({"error": error}, status_code=400)
            source = f"uploaded clip ({clip.filename})"
        elif text and text.strip():
            query = demo.embedder.embed_text([text.strip()])[0]
            source = f'text: "{text.strip()}"'
        else:
            return JSONResponse({"error": "type a phrase or choose a clip"},
                                status_code=400)
        return {"query": source, "results": demo.cards(demo.search(query))}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="store/hnsw_locality_large")
    ap.add_argument("--db", default="store/large/timbre.db")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-warm", action="store_true")
    a = ap.parse_args()

    import uvicorn

    print(f"Loading {a.graph}...", flush=True)
    demo = Demo(a.graph, a.db, a.device)
    print(f"{demo.index.size:,} indexed segments, {len(demo.tracks):,} tracks", flush=True)
    if demo.model_error:
        print(f"WARNING: model unavailable: {demo.model_error}", flush=True)
    if demo.version_drift:
        print("NOTE: embedding environment differs from the store's", flush=True)
    if not a.no_warm:
        demo.warm()
        print("Index warmed.", flush=True)
    uvicorn.run(create_app(demo), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    raise SystemExit(main())
