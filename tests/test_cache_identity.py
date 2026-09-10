"""A model migration must invalidate old nearest-neighbor caches."""
import sqlite3

import numpy as np
import pytest

from timbre.groundtruth import load_groundtruth, source_identity
from timbre.manifest import connect


@pytest.fixture
def source(tmp_path):
    db, store, cache = (tmp_path / name for name in ("t.db", "v.npy", "gt.npz"))
    conn = connect(db)
    conn.executemany("INSERT INTO meta VALUES (?, ?)",
                     [("model", "model-a"), ("model_revision", "revision-a")])
    conn.execute("INSERT INTO tracks (track_id,path,row_start,n_windows,status,genre) "
                 "VALUES (1,'test.mp3',0,1,'done','Jazz')")
    conn.commit()
    conn.close()
    np.save(store, np.ones((1, 512), dtype=np.float32))
    np.savez(cache, gt_ids=np.array([[0]]), **source_identity(db, store))
    return db, store, cache


def test_matching_cache_loads(source):
    db, store, cache = source
    np.testing.assert_array_equal(load_groundtruth(cache, db, store)["gt_ids"], [[0]])


@pytest.mark.parametrize("change", ["vectors", "model", "genre", "legacy"])
def test_cache_rejects_changed_source(source, change):
    db, store, cache = source
    if change == "vectors":
        np.save(store, np.zeros((1, 512), dtype=np.float32))
    elif change == "legacy":
        np.savez(cache, gt_ids=np.array([[0]]))
    else:
        with sqlite3.connect(db) as conn:
            if change == "model":
                conn.execute("UPDATE meta SET value='model-b' WHERE key='model'")
            else:
                conn.execute("UPDATE tracks SET genre='Rock'")
    with pytest.raises(RuntimeError, match="stale"):
        load_groundtruth(cache, db, store)


def test_cache_rejects_incomplete_ingest(source):
    db, store, _ = source
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE tracks SET status='pending'")
    with pytest.raises(RuntimeError, match="incomplete"):
        source_identity(db, store)
