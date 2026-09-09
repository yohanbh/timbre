"""Fast regression tests for the invariants that make restarts byte-identical."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
from timbre.audio import windows
from timbre.embed import WINDOW_SAMPLES, WINDOWS_PER_TRACK, Embedder, assert_window_len


def test_window_length_invariant():
    """480001 samples would silently trigger unseeded rand_trunc cropping."""
    assert_window_len(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    with pytest.raises(ValueError):
        assert_window_len(np.zeros(WINDOW_SAMPLES + 1, dtype=np.float32))


def test_windowing_counts():
    assert len(windows(np.zeros(48000 * 30, dtype=np.float32))) == WINDOWS_PER_TRACK
    assert len(windows(np.zeros(48000 * 12, dtype=np.float32))) == 3
    assert len(windows(np.zeros(48000 * 9, dtype=np.float32))) == 0


@pytest.mark.slow
def test_embedding_is_bit_reproducible():
    """Same batch twice must be bit-identical -- the core of the phase gate."""
    emb = Embedder()
    rng = np.random.default_rng(0)
    wins = [rng.standard_normal(WINDOW_SAMPLES).astype(np.float32)
            for _ in range(WINDOWS_PER_TRACK)]
    assert np.array_equal(emb.embed(wins), emb.embed(wins))


@pytest.mark.slow
def test_crash_window_rewrites_identical_bytes(tmp_path):
    """A crash between fsync and the done-marker commit must be harmless.

    Rolls back only the marker, leaving vectors on disk -- exactly the unsafe-looking
    window -- then re-runs. The rewritten bytes must be identical, which is what
    makes byte-identity survive arbitrary kill points.
    """
    import shutil
    import sqlite3
    import subprocess

    root = tmp_path / "audio" / "000"
    root.mkdir(parents=True)
    for i in (2, 3):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi",
             "-i", f"sine=frequency={200 + i * 17}:duration=30:sample_rate=44100",
             "-ac", "2", "-b:a", "128k", str(root / f"{i:06d}.mp3"), "-y"],
            check=True,
        )

    from timbre.ingest import run
    db, store = str(tmp_path / "t.db"), str(tmp_path / "v.npy")
    run(str(tmp_path / "audio"), db, store)

    conn = sqlite3.connect(db)
    tid, row_start, n = conn.execute(
        "SELECT track_id, row_start, n_windows FROM tracks WHERE status='done' LIMIT 1"
    ).fetchone()
    before = np.array(np.load(store, mmap_mode="r")[row_start:row_start + n])

    conn.execute(
        "UPDATE tracks SET status='pending', n_windows=NULL WHERE track_id=?", (tid,)
    )
    conn.commit()
    conn.close()

    run(str(tmp_path / "audio"), db, store)
    after = np.array(np.load(store, mmap_mode="r")[row_start:row_start + n])
    assert np.array_equal(before, after)
