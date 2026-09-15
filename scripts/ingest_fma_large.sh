#!/usr/bin/env bash
# Supervised, resumable download -> verify -> extract -> reuse -> embed -> verify.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p store/large
exec 9>store/large/run.lock
flock -n 9 || { echo 'Another FMA large pipeline is running'; exit 1; }
export PYTHONPATH=src PYTHONUNBUFFERED=1 USE_TF=0 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

stage=starting
set_stage() {
    stage=$1
    printf '%s\n' "$stage" >store/large/stage
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$stage"
}
trap 'code=$?; if (( code != 0 )); then printf "failed:%s (exit %s)\n" "$stage" "$code" >store/large/stage; fi' EXIT

if [[ -f store/large/extraction.paused ]]; then
    set_stage extraction_paused
    echo 'Extraction is paused. Wait for explicit user instruction before removing store/large/extraction.paused.'
    exit 0
fi

archive=data/fma_large.zip
expected_sha1=497109f4dd721066b5ce5e5f250ec604dc78939e
download_unit=timbre-fma-large-download.service
if [[ -f store/large/extracted.sha1 ]]; then
    set_stage verifying_extraction
elif [[ ! -f "$archive" ]]; then
    set_stage downloading
    if [[ $(systemctl --user show "$download_unit" -p LoadState --value) == not-found ]]; then
        systemd-run --user --unit=timbre-fma-large-download \
            --description='Download official FMA large archive' \
            --property="WorkingDirectory=$PWD" --property=RemainAfterExit=yes \
            --property=Restart=on-failure --property=RestartSec=30 \
            --property="StandardOutput=append:$PWD/data/fma_large_download.log" \
            --property="StandardError=append:$PWD/data/fma_large_download.log" \
            /usr/bin/curl --fail --location --silent --show-error --continue-at - \
            --retry 20 --retry-delay 10 --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
            --output "$PWD/$archive.part" https://os.unil.cloud.switch.ch/fma/fma_large.zip
    else
        systemctl --user start "$download_unit"
    fi
    while [[ $(systemctl --user show "$download_unit" -p SubState --value) != exited ]]; do
        if [[ $(systemctl --user show "$download_unit" -p ActiveState --value) == failed ]]; then
            echo 'Download service failed; see data/fma_large_download.log' >&2
            exit 1
        fi
        sleep 15
    done
    test "$(systemctl --user show "$download_unit" -p ExecMainStatus --value)" = 0
    set_stage verifying_archive
    printf '%s  %s\n' "$expected_sha1" "$archive.part" | sha1sum --check -
    mv "$archive.part" "$archive"
else
    set_stage verifying_archive
    printf '%s  %s\n' "$expected_sha1" "$archive" | sha1sum --check -
fi

if [[ ! -f store/large/extracted.sha1 ]]; then
    set_stage extracting
fi
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import subprocess
import zipfile
from timbre.metadata import read_csv

expected = {row[0] for row in read_csv('data/fma_metadata/tracks.csv')}
assert len(expected) == 106574, 'unexpected metadata corpus size'
root, marker = Path('data/fma_large'), Path('store/large/extracted.sha1')
checksum = '497109f4dd721066b5ce5e5f250ec604dc78939e'
def complete():
    tracks = [int(path.stem) for path in root.rglob('*.mp3')]
    return len(tracks) == len(expected) and set(tracks) == expected
if marker.exists():
    assert marker.read_text().strip() == checksum, 'invalid extraction marker'
    assert complete(), 'extracted corpus is incomplete; restore missing tracks before resuming'
else:
    with zipfile.ZipFile('data/fma_large.zip') as archive:
        names = archive.namelist()
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == 'fma_large', name
        tracks = [int(PurePosixPath(name).stem) for name in names if name.endswith('.mp3')]
        assert len(tracks) == len(expected) and set(tracks) == expected, 'archive membership mismatch'
    subprocess.run(['unzip', '-oq', 'data/fma_large.zip', '-d', 'data'], check=True)
    assert complete(), 'extraction did not produce the full corpus'
    marker.write_text(checksum + '\n')
print('Verified all 106,574 extracted track IDs', flush=True)
PY

if [[ -f store/large/vectorization.paused ]]; then
    set_stage awaiting_vectorization_approval
    echo 'Vectorization is paused. Wait for explicit user instruction before removing store/large/vectorization.paused.'
    exit 0
fi

set_stage reusing_medium_vectors
python3 - <<'PY'
import json
from pathlib import Path
from timbre import manifest, metadata
from timbre.groundtruth import load_groundtruth
from timbre.seed_store import seed

source = load_groundtruth('store/groundtruth.npz', 'store/timbre.db', 'store/vectors.npy')
conn, count = manifest.build('data/fma_large', 'store/large/timbre.db', 'store/large/vectors.npy')
assert count == 106574
metadata.load(conn, 'data/fma_metadata/tracks.csv')
conn.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',
             ('seed_source_store_sha256', str(source['source_store_sha256'])))
conn.commit()
conn.close()
summary = seed('store/timbre.db', 'store/vectors.npy', 'store/large/timbre.db', 'store/large/vectors.npy')
Path('store/large/seed_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(summary, flush=True)
PY

set_stage embedding
python3 -u -m timbre --audio-root data/fma_large \
    --db store/large/timbre.db --store store/large/vectors.npy

set_stage verifying_vectors
python3 - <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import numpy as np
from timbre.verify import check_holes_and_norms, hash_db, hash_store

db, path = 'store/large/timbre.db', 'store/large/vectors.npy'
with sqlite3.connect(db) as conn:
    counts = dict(conn.execute('SELECT status,COUNT(*) FROM tracks GROUP BY status'))
    populated = conn.execute("SELECT COALESCE(SUM(n_windows),0) FROM tracks WHERE status='done'").fetchone()[0]
    meta = dict(conn.execute('SELECT key,value FROM meta'))
assert sum(counts.values()) == 106574 and counts.get('pending', 0) == 0, counts
vectors = np.load(path, mmap_mode='r')
assert vectors.shape == (106574 * 21, 512) and vectors.dtype == np.float32
holes, norms = check_holes_and_norms(db, path)
assert not holes and not norms, (holes[:5], norms[:5])
summary = {'completed_at': datetime.now(timezone.utc).isoformat(), 'status_counts': counts,
           'populated_vectors': populated, 'allocated_shape': list(vectors.shape),
           'store_sha256': hash_store(path), 'db_sha256': hash_db(db),
           'model': meta['model'], 'model_revision': meta['model_revision'],
           'seeded_tracks': int(meta.get('seeded_tracks', 0)), 'holes_ok': True, 'norms_ok': True}
Path('store/large/completion.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
PY
set_stage complete
