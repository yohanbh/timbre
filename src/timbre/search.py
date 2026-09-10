"""Exact segment search, returning each track's best matching passage."""
import numpy as np

from .groundtruth import CHUNK_ROWS


def exact_track_topk(store, query, row_ids, owner, k=5, exclude_track=None,
                     chunk=CHUNK_ROWS):
    """Max over each track's real windows; return (track_ids, row_ids, scores).

    `row_ids` and `owner` come from load_layout, so unpopulated store rows are
    never candidates. Equal scores are resolved by the smaller window row ID.
    """
    if k < 1 or chunk < 1:
        raise ValueError("k and chunk must be positive")
    query = np.asarray(query, dtype=np.float32)
    if query.shape != (store.shape[1],) or not np.isfinite(query).all():
        raise ValueError("query must be a finite vector matching the store dimension")
    norm = np.linalg.norm(query)
    if norm == 0:
        raise ValueError("query must be nonzero")
    query = query / norm
    scores = np.empty(len(row_ids), dtype=np.float32)
    for start in range(0, len(row_ids), chunk):
        sub = row_ids[start:start + chunk]
        scores[start:start + len(sub)] = np.asarray(store[sub]) @ query
    if exclude_track is not None:
        scores[owner == exclude_track] = -np.inf

    order = np.lexsort((row_ids, -scores))
    seen, hits = set(), []
    for i in order:
        if not np.isfinite(scores[i]):
            break
        if owner[i] in seen:
            continue
        seen.add(owner[i])
        hits.append(i)
        if len(hits) == k:
            break
    hits = np.asarray(hits, dtype=np.int64)
    return owner[hits], row_ids[hits], scores[hits]
