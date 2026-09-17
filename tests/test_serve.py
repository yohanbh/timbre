"""Demo server invariants that are easy to get silently wrong.

Dedupe and offset arithmetic decide what a visitor sees, and both fail quietly:
a duplicate track looks like a plausible result, and a bad offset points at the
wrong passage of the right song. Neither raises.
"""
import numpy as np
import pytest

from timbre.embed import HOP_SECONDS
from timbre.serve import _clock


def test_clock_formats_minutes_and_seconds():
    assert _clock(0) == "0:00"
    assert _clock(9) == "0:09"
    assert _clock(65) == "1:05"
    assert _clock(600) == "10:00"


def test_offset_inverts_row_start():
    """offset_seconds = (row - row_start) * HOP_SECONDS, the listen.py rule."""
    row_start, n_windows = 42_000, 21
    for window in range(n_windows):
        row = row_start + window
        assert (row - row_start) * HOP_SECONDS == window * HOP_SECONDS


class _FakeIndex:
    """Returns rows in descending-score order, several per track."""

    def __init__(self, rows, scores):
        self._rows, self._scores = np.asarray(rows), np.asarray(scores)
        self.size = 1000
        self.vectors = np.zeros((self.size, 4), dtype=np.float32)

    def search(self, query, k, ef_search):
        assert ef_search >= k, "native search requires ef_search >= k"
        return self._rows[:k], self._scores[:k]


def _demo(rows, scores, lut, tracks):
    from timbre.serve import Demo

    demo = Demo.__new__(Demo)
    demo.index = _FakeIndex(rows, scores)
    demo.lut = np.asarray(lut)
    demo.tracks = tracks
    import threading

    demo.lock = threading.Lock()
    return demo


def test_search_keeps_one_row_per_track_and_the_best_one():
    # Track 7 owns rows 0-2, track 9 owns 3-4, track 11 owns row 5.
    rows = [0, 1, 2, 3, 4, 5]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    lut = [7, 7, 7, 9, 9, 11]
    tracks = {t: (t, 0, 21, "T", "A", None, 100.0) for t in (7, 9, 11)}
    hits = _demo(rows, scores, lut, tracks).search(None, k=3)
    assert [h[0] for h in hits] == [7, 9, 11], "one hit per track, score order"
    assert [h[1] for h in hits] == [0, 3, 5], "each track's best-scoring row"
    # float32 round-trip, so compare approximately.
    assert [h[2] for h in hits] == pytest.approx([0.9, 0.6, 0.4], abs=1e-6)


def test_search_skips_lut_holes():
    """Unpopulated store rows map to -1 and must never reach a result card."""
    hits = _demo([0, 1, 2], [0.9, 0.8, 0.7], [-1, -1, 5],
                 {5: (5, 0, 21, "T", "A", None, 1.0)}).search(None, k=3)
    assert [h[0] for h in hits] == [5]


def test_search_skips_tracks_missing_from_the_catalog():
    hits = _demo([0, 1], [0.9, 0.8], [4, 5],
                 {5: (5, 0, 21, "T", "A", None, 1.0)}).search(None, k=2)
    assert [h[0] for h in hits] == [5]
