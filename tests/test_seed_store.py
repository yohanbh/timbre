"""Reuse must remap track rows and never copy an embedding for different audio."""
import numpy as np
import pytest

from timbre import manifest
from timbre.seed_store import seed


def make_store(tmp_path, name, tracks):
    root = tmp_path / name
    root.mkdir()
    for tid, contents in tracks.items():
        (root / f"{tid:06d}.mp3").write_bytes(contents)
    db, vectors = tmp_path / f"{name}.db", tmp_path / f"{name}.npy"
    conn, _ = manifest.build(root, db, vectors)
    return conn, db, vectors


def test_reuse_remaps_rows_preserves_padding_and_skips_changed_audio(tmp_path):
    src, src_db, src_path = make_store(tmp_path, "source", {10: b"same", 30: b"old"})
    dst, dst_db, dst_path = make_store(tmp_path, "target", {2: b"new", 10: b"same", 20: b"new", 30: b"changed"})
    data = np.load(src_path, mmap_mode="r+")
    data[:20, 0] = 1
    data[21:41, 1] = 1
    data.flush()
    src.execute("UPDATE tracks SET status='done', n_windows=20")
    src.commit()
    before = src_path.read_bytes()
    result = seed(src_db, src_path, dst_db, dst_path)
    assert result == {"copied_this_run": 1, "seeded_tracks_total": 1, "audio_mismatches": 1}
    actual = np.load(dst_path)
    np.testing.assert_array_equal(actual[21:42], data[:21])
    assert not actual[:21].any() and not actual[42:].any()
    assert dict(dst.execute("SELECT track_id,status FROM tracks")) == {2: "pending", 10: "done", 20: "pending", 30: "pending"}
    assert seed(src_db, src_path, dst_db, dst_path)["copied_this_run"] == 0
    assert src_path.read_bytes() == before
    src.close()
    dst.close()


def test_reuse_rejects_embedding_configuration_drift(tmp_path):
    src, src_db, src_path = make_store(tmp_path, "source", {10: b"same"})
    dst, dst_db, dst_path = make_store(tmp_path, "target", {10: b"same"})
    src.execute("UPDATE meta SET value='different' WHERE key='model_revision'")
    src.commit()
    with pytest.raises(RuntimeError, match="environment changed"):
        seed(src_db, src_path, dst_db, dst_path)
    assert not np.load(dst_path).any()
    src.close()
    dst.close()
