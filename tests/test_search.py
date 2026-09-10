"""Track search must play the winning segment and exclude siblings and holes."""
import numpy as np
import pytest

from timbre.search import exact_track_topk


def test_best_passage_per_track_across_chunks():
    # Track 20's good match is its second window. Averaging would rank track 30
    # first. Rows 2 and 5 are unused allocations and must never be candidates.
    store = np.array([[1, 0], [1, 0], [0, 0], [-1, 0], [1, 0],
                      [0, 0], [0.8, 0.6], [0.8, 0.6], [-1, 0]], dtype=np.float32)
    rows = np.array([0, 1, 3, 4, 6, 7, 8])
    owners = np.array([10, 10, 20, 20, 30, 30, 40])
    ids, hits, scores = exact_track_topk(
        store, store[0], rows, owners, exclude_track=10, k=10, chunk=2
    )
    np.testing.assert_array_equal(ids, [20, 30, 40])
    np.testing.assert_array_equal(hits, [4, 6, 8])
    np.testing.assert_allclose(scores, [1, 0.8, -1])


@pytest.mark.parametrize("query", [[0, 0], [np.nan, 0], [1, 0, 0]])
def test_invalid_query_rejected(query):
    with pytest.raises(ValueError):
        exact_track_topk(np.eye(2), query, np.array([0, 1]), np.array([10, 20]))
