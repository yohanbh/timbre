# Next steps

Phases 0 and 1 are implemented. The checkpoint migration is complete; Phase 2
has not started. See [RETRIEVAL_DIAGNOSIS.md](RETRIEVAL_DIAGNOSIS.md) for the failure
analysis, controlled comparison, and repair validation.

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
The completed store passes integrity checks and all 26 non-slow tests. The three
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

## Phase 2 — HNSW from scratch

Implement exponential layer assignment, greedy descent, beam search at layer 0,
insertion with bidirectional links and pruning, the neighbor-selection heuristic,
and graph serialization. Do not substitute a third-party index in the serving path.

Done when recall@10 is at least 0.95 against Phase 1 ground truth, with an
`M × efConstruction × efSearch` sweep showing the recall/latency frontier.

Re-measure geometric properties on the replacement embeddings before drawing
conclusions from the old checkpoint's anisotropy. Still-open experiments include
max versus mean versus count aggregation, shuffled insertion order, and human
evaluation of text/audio retrieval. The user's concrete listening examples are
tracks 2, 3, and 10, including the fifth neighbor of track 10.
