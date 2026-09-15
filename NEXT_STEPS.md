# Next steps

Phases 0, 1 and 2 are implemented. The checkpoint migration and full-medium
baseline are complete; both HNSW benchmark sizes pass the recall gate.
Phase 3's C++ query traversal and graph construction are implemented and benchmarked;
see [PHASE3.md](PHASE3.md). The user authorized FMA large expansion on 2026-09-10:
download, extraction, vectorization and final verification are complete.
The large store contains 2,144,867 populated vectors from 105,884 embedded tracks,
with 162 failed, 528 too short and zero pending. Windows disk compaction also
completed, reclaiming 110.60 GiB. See [FMA_LARGE.md](FMA_LARGE.md).
The remaining scale and memory experiments are still open.
See [RETRIEVAL_DIAGNOSIS.md](RETRIEVAL_DIAGNOSIS.md)
for the retrieval repair and [PHASE2.md](PHASE2.md) for index implementation and results.

## Current store

The active model is `laion/larger_clap_general`, pinned with its processor to
`ada0c23a36c4e8582805bb38fec3905903f18b41`. The original music checkpoint collapsed
unrelated text queries and produced weak audio retrieval.

| Artifact | Current state |
|---|---|
| `store/vectors.npy` | 510,064 real vectors, 512-dimensional float32 |
| `store/timbre.db` | 24,980 embedded tracks; 15 corrupt and five too short |
| `store/groundtruth.npz` | Exact top-100 for 1,000 fixed, genre-stratified queries |
| `store/legacy_music/` | Original vectors, database, cache and ingest log |

New vector SHA-256:
`67bfc23b7f26fe97c0751192be478ebfea50464462c334ed44cbc9af7026d3da`.
The completed store passes integrity checks and the non-slow test suite. The three
GPU tests passed during the rebuild, including determinism and crash-window
recovery. The original full-scale SIGKILL experiment remains historical; it was
not repeated on the entire replacement store.

On the same 2,000 queries against all 24,980 tracks, track-mean genre@10 rose
from **31.5% to 70.0%** (chance 17.2%). Centering gives 70.2%, so raw embeddings
remain the default. This is a relevance proxy, separate from human listening.

## Querying and validation

```bash
# Play the actual query and winning ten-second passage from each result.
PYTHONPATH=src python3 tests/listen.py --track 2 -k 5 --play
PYTHONPATH=src python3 tests/listen.py --track 10 --offset 10 -k 5 --play
PYTHONPATH=src python3 tests/listen.py --text "sparse melancholy piano" -k 5 --play

# Store integrity, then the complete test suite (includes GPU tests).
USE_TF=0 PYTHONPATH=src python3 -m timbre.verify
USE_TF=0 PYTHONPATH=src python3 -m pytest -q

# Reproduce full-corpus genre measurement.
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 tests/centering_test.py 2000

# Rebuild ground truth if the source vectors or manifest change.
USE_TF=0 PYTHONPATH=src python3 -m timbre.build_groundtruth
```

Listening searches a single segment and ranks distinct tracks by their best
matching segment. It excludes the query's whole track and plays each winner at
the matching offset. Text queries reject a store/model configuration mismatch.
Ingest checks for text-embedding collapse before a long run.

Use `python -m timbre` for ingest: it sets thread limits before importing NumPy
or PyTorch. A model change requires a fresh database and vector store; the
existing resume guard correctly rejects mixed embedding configurations. There
is no pending work in the active store.

## Ground truth for Phase 2

Use the validated loader before scoring an index:

```python
from timbre.groundtruth import load_groundtruth, recall_at_k

truth = load_groundtruth("store/groundtruth.npz", "store/timbre.db", "store/vectors.npy")
# recall_at_k(candidate_ids, truth["gt_ids"], 10)
```

The cache records hashes of vectors and the manifest, plus the model revision.
Loading rejects stale caches; building rejects incomplete ingests. Four BLAS
threads are pinned when building the cache to fix float32 reduction behavior.

| Arrays | Purpose |
|---|---|
| `gt_ids`, `gt_sims` | Geometric index recall, excluding the query row itself |
| `gt_ids_nosib`, `gt_sims_nosib` | Cross-track geometric recall, excluding all query-track windows |

With the replacement model, **94.92%** of unfiltered top-10 hits are same-track
siblings, and **81.5%** of queries have an all-sibling top-10. Adjacent windows
overlap by 90%; cross-track relevance must be evaluated separately. Neither
cache contains human judgments of musical similarity.

All candidate row IDs must come from `load_layout`; the mmap reserves 21 rows
per track but most tracks fill 20, leaving zero-filled holes. Segment offsets
are derivable from row IDs and each track's row start.

## Phase 2 — complete

`src/timbre/hnsw.py` implements exponential layers, greedy descent, beam search,
diversified neighbor selection, reciprocal insertion with bounded pruning, and
graph serialization. Graph reload verifies the exact vector fingerprint; saved
random-generator state supports continued insertion. No third-party ANN code is
used by the implementation.

The spec's Phase 2 benchmark uses a reproducible 125,000-vector subset, with all
1,000 query rows held out from construction. `benchmark_hnsw` recomputes the exact
oracle over that candidate set; the full-corpus cache above is not a valid oracle
for a subset graph. Artifacts live in `store/hnsw_phase2`, and all 20 measured
parameter combinations are recorded in `docs/hnsw_results.json`.

Recommended measured setting: **M=8, efConstruction=80, efSearch=64**. It achieved
99.43% recall@10, 0.984 ms median and 1.179 ms p95 query latency, on one CPU thread.
Build time was 189 seconds; the serialized graph is 8.89 MiB, excluding vectors.
Re-loading the 125,000-node graph reproduces the measured recall.

```bash
PYTHONPATH=src python3 -m timbre.benchmark_hnsw
USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest -q -m 'not slow'
```

## Phase 3 — native implementation and corpus expansion complete; scale experiments remain

Per the spec, Phase 3 covers the C++ distance/search loop, larger corpus, and memory
limit experiments. Native query traversal is complete. On the same saved graphs,
at efSearch=64, it measured **4.35× faster on 125K** and **4.55× faster on medium**
than freshly measured Python. Recall@10 remains 99.43% and 99.85% respectively;
all top-10 neighbor sets match. C++ medium latency is 0.193 ms median and 0.267 ms
p95. See [PHASE3.md](PHASE3.md) for all settings, setup and reproduction commands.

Native construction is now complete. Fresh paired builds took **195.44 s versus
36.99 s on 125K (5.28× faster)** and **826.95 s versus 170.76 s on medium (4.84×)**.
These totals include initialization, checkpoint writes and final saving. Recall
was unchanged at every tested efSearch setting. Reloading the native checkpoints
reproduced 99.43% and 99.85% recall@10 at efSearch=64.

Artifacts are in `store/hnsw_construction_125k` and
`store/hnsw_construction_medium`; report copies are in `docs`. Construction
supports deterministic checkpoint continuation with the remaining original
insertion order and the same native backend. See [PHASE3.md](PHASE3.md) for
reproduction commands and timing details.

FMA large ingestion and final verification completed on 2026-09-10 at 22:23 EDT.
All 24,980 matching medium tracks were reused. Final counts are 105,884 embedded,
162 failed, 528 short and zero pending, with 2,144,867 populated vectors.
`store/large/completion.json` records the verified shape, norms and hashes.
See [FMA_LARGE.md](FMA_LARGE.md) for completion and historical recovery details.

Large-store ground truth and the held-out benchmark oracle are complete and
validated. `store/large/groundtruth.npz` contains both full-store variants for
1,000 fixed queries; `store/large/index_groundtruth.npz` contains the exact
answers over 2,143,867 candidates with all query rows held out. See
[the validation report](docs/large_groundtruth_results.json).

Completed: large ground-truth artifacts are ready and no re-embedding is required.

Running: start held-out large native construction and search under `store/hnsw_construction_large`
and `store/hnsw_native_large` with the commands below. Track exact command,
git hash and output files at each run to preserve reproducibility.

```bash
PYTHONPATH=src python3 -m timbre.benchmark_construction \
  --baseline store/hnsw_medium --out store/hnsw_construction_large \
  --oracle store/large/index_groundtruth.npz --db store/large/timbre.db \
  --store store/large/vectors.npy --groundtruth store/large/groundtruth.npz \
  --backends cpp --m 8 --ef-construction 80 --ef-search 16 32 64 128 256

PYTHONPATH=src python3 -m timbre.benchmark_native \
  --baseline store/hnsw_medium --out store/hnsw_native_large/results.json \
  --oracle store/large/index_groundtruth.npz --graph store/hnsw_construction_large/cpp_direct \
  --db store/large/timbre.db --store store/large/vectors.npy \
  --groundtruth store/large/groundtruth.npz --m 8 --ef-construction 80 \
  --ef-search 16 32 64 128 256
```

Next: add recall@100 runs with `--k 100` and `--ef-search` values >= 100 using the
saved large graph path, then capture recall/latency gates and move to direct
loading under memory caps.
Open issues: direct mmap loading, memory-cap tables and locality reordering are
not yet implemented.

The completed full-medium **Python** baseline remains available for comparison:

The graph contains **509,064 candidate segments**, with all 1,000 fixed query
rows held out. At **M=8, efConstruction=80, efSearch=64**, it reaches **99.85%
recall@10, 1.139 ms median and 1.835 ms p95**. Construction took 862.7 seconds;
the graph serializes to 34.89 MiB, excluding vectors. All five efSearch settings
passed the recall gate, and reloading reproduced the efSearch=64 result.

Artifacts and the recomputed exact oracle are in `store/hnsw_medium`.
See the [full-medium results and validation](PHASE2.md#full-medium-baseline-2026-09-10).
Reproduce with:

```bash
PYTHONPATH=src python3 -m timbre.benchmark_hnsw \
  --vectors 509064 --queries 1000 --m 8 --ef-construction 80 \
  --ef-search 16 32 64 128 256 --out store/hnsw_medium
```

Still-open experiments include max versus mean versus count aggregation,
insertion order (`--order track` versus the default shuffled order), third-party
benchmark baselines, and human evaluation of text/audio retrieval. The user's
concrete listening examples are tracks 2, 3 and 10, including the fifth neighbor
of track 10. Keep musical relevance separate from geometric index recall.
