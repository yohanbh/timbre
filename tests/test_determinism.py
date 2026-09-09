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
