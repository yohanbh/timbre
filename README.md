# Timbre

Audio similarity search over the Free Music Archive, on a hand-written HNSW index.
See `TIMBRE_SPEC.md` for scope. **The index is the project**; Phase 0 is the pipeline
that feeds it.

The embedding model is pinned to `laion/larger_clap_general` at
`ada0c23a36c4e8582805bb38fec3905903f18b41`. The original music checkpoint
produced collapsed text embeddings and weak audio retrieval; see
[the diagnosis and controlled comparison](RETRIEVAL_DIAGNOSIS.md).

The full rebuild is active at `store/timbre.db`, `store/vectors.npy`, and
`store/groundtruth.npz`; the original files are archived in `store/legacy_music`.
On the same 2,000 queries against all 24,980 tracks, track-mean genre@10 improved
from **31.5% to 70.0%** (chance 17.2%). This holds the aggregation rule constant
to measure the checkpoint change; genre agreement is a proxy, not a listening
judgment. The listening tool separately uses matching segments as described below.

## Phase 2 — HNSW

The hand-written Python/NumPy index passes the Phase 2 gate on **125,000 indexed
segments and 1,000 held-out queries**. The 20-point parameter sweep uses exact
neighbors recomputed over that same candidate set.

| M | efConstruction | efSearch | Recall@10 | Median | p95 |
|---|---|---|---|---|---|
| 8 | 80 | 16 | 95.72% | 0.386 ms | 0.524 ms |
| 8 | 80 | 32 | 98.36% | 0.581 ms | 0.760 ms |
| 8 | 80 | 64 | **99.43%** | **0.984 ms** | **1.179 ms** |
| 16 | 160 | 64 | 99.85% | 1.650 ms | 2.460 ms |

Measured on an AMD Ryzen 7 5800H, with one CPU/BLAS thread and warm caches.
The `M=8, efConstruction=80` graph took 189 seconds to build and 8.89 MiB to
serialize, excluding vectors. Re-loading it reproduces the 99.43% recall result.
These timings describe the 125k benchmark, not the complete 510k-vector store.

![HNSW recall versus query latency](docs/hnsw_frontier.svg)

See [implementation and benchmark protocol](PHASE2.md) and
[all measured results](docs/hnsw_results.json). Run:

```bash
PYTHONPATH=src python3 -m timbre.benchmark_hnsw
```

## Retrieval and validation

```bash
# Search a particular passage; play the matching passage from each distinct track.
PYTHONPATH=src python3 tests/listen.py --track 2 --offset 10 -k 5 --play
PYTHONPATH=src python3 tests/listen.py --text "sparse melancholy piano" -k 5 --play

# Rebuild exact segment neighbors after any vector-store/model change.
PYTHONPATH=src python3 -m timbre.build_groundtruth
USE_TF=0 PYTHONPATH=src python3 -m pytest -q
```

Listening uses each track's **best segment score**, and plays that segment's
actual offset. The query's entire track is excluded for audio queries. Text
queries are rejected if the store's embedding configuration differs from the
query model. Ingest checks for collapsed text embeddings before starting.

Ground truth measures geometric index recall, not human judgments of musical
similarity. Both unfiltered and sibling-excluded top-100 lists are cached. Load
them with the source-validation helper so an old cache cannot silently be used
after a model or data change:

```python
from timbre.groundtruth import load_groundtruth, recall_at_k

truth = load_groundtruth("store/groundtruth.npz", "store/timbre.db", "store/vectors.npy")
# recall_at_k(candidate_ids, truth["gt_ids"], 10)
```

Cache validation checks hashes of the vectors and the manifest (including model
revision and genre labels). The builder refuses an incomplete ingest. The
existing resume check refuses to mix different embedding configurations in one
store; rebuild into a separate directory when changing models.

## Phase 0 — Ingest pipeline

```
mp3 ──► ffmpeg decode ──► 21 windows ──► CLAP ──► mmap .npy
        48 kHz mono       10 s @ 1 s hop  512-d    + SQLite
```

```bash
uv sync
python -m timbre --audio-root data/fma_medium          # resumable; just rerun after a crash
python -m timbre.verify                                 # hashes + integrity checks
bash tests/kill_restart.sh <audio_root> <db> <store> 5  # the acceptance gate
```

### Original pipeline measurements (RTX 3060 6 GB, 16 cores, 7.4 GiB RAM)

| Stage | Cost per track |
|---|---|
| ffmpeg decode | 113 ms |
| Mel extraction (1 core) | 1000 ms |
| CLAP forward, batch of 21 | 159 ms (132 windows/s) |

With 12 workers the CPU pool supplies 10.8 tracks/s against 6.3 tracks/s of GPU
capacity, so the pipeline is GPU-bound by design — the irreducible cost. Without
the pool a single CPU feeds only 0.9 tracks/s and the GPU idles ~85%. Peak VRAM
1475 MiB. Measured full-corpus run: **25,000 tracks in ~1.4 h at 5.0 tracks/s**,
GPU pegged at 91-97%, ~1.6 GiB RAM free throughout.

### Thread oversubscription (the bug that cost the most)

Setting `OMP_NUM_THREADS=1` inside the worker initializer is a **no-op**: BLAS and
OpenMP size their thread pools when numpy/torch are *imported*, which for a forked
worker has already happened. Each worker started ~17 threads; 6 workers put ~100
runnable threads on 16 cores, and throughput collapsed to **0.83 tracks/s** (8.4 h
ETA) with the GPU idling at 8% and CPU at 98.5% user.

The fix is `src/timbre/__main__.py`, which sets the limits before importing
anything heavy. After: 2 threads/worker, 5.0 tracks/s, GPU 96%. Run as
`python -m timbre`, not `python -m timbre.ingest`.

The diagnostic signature worth remembering: **high user CPU with an idle GPU means
contention, not starvation.** Check thread counts before tuning worker counts.

### Backpressure

`ProcessPoolExecutor.map` submits every task at once, so workers race ahead of the
GPU consumer and completed mel tensors accumulate without bound — 5.1 MiB each,
~125 GiB across 25k tracks on a 7.4 GiB machine. The loop instead keeps at most
`QUEUE_DEPTH` futures alive, topping up as results are consumed, which caps
in-flight mel at ~123 MiB for any corpus size.

That change also happens to prove the order-independence claim: `wait(FIRST_COMPLETED)`
yields results in completion order where `map` yielded submission order, and the
store hash is unchanged across both. Row assignment genuinely does not depend on
processing order.

### What "byte-identical restart" actually requires

The phase gate is: kill the job anywhere, restart, get a bit-for-bit identical
store. Three findings, each measured rather than assumed:

**1. Batch composition changes the output bits.** The same window embedded inside
a differently-sized batch differs by ~5e-08:

| Test | Result |
|---|---|
| Same batch twice | identical |
| `cudnn.benchmark` on vs off | identical |
| Window 0 alone vs inside a batch of 21 | **differs**, 3.8e-08 |
| Batch of 5 vs the same 5 inside a batch of 21 | **differs**, 5.0e-08 |
| TF32 on vs off | **differs**, 1.0e-04 |

Batched GEMM kernels select tile decompositions by batch shape, so accumulation
order — and the low bits — change. Fix: **one track = one batch of exactly 21
windows**, making composition a pure function of track ID.

**2. `truncation: "rand_trunc"` is live in the shipped preprocessor config.** The
random-crop branch fires on `len > 480000`, *strictly* greater, calling an
unseeded `np.random.randint`. Windows of exactly 480,000 samples stay on the
deterministic path; an off-by-one would surface only as an intermittently failing
hash hours into a run. Asserted in `embed.assert_window_len`.

**3. Write ordering must be write → flush → fsync → *then* commit.** The reverse
lets a crash between the commit and the fsync leave a track marked `done` whose
rows are permanently zeroed — silent corruption, since the job reports success.
With this ordering the worst case is redundant re-embedding that rewrites
identical bytes over the same rows.

Bit-identity is promised **within a fixed software and hardware configuration**.
Library versions and the model commit SHA are recorded in the `meta` table, and a
resume against a changed environment is refused rather than silently producing
different bytes.

### Corpus reality: 20 windows, not 21

The spec sizes the index at 21 windows/track from 30 s clips. Measured on the real
corpus, **59% of fma_medium decodes to 29.9766 s** -- 23 ms short of 30.0 s, an mp3
encoder artifact rather than genuinely short music (zero sampled tracks were short
by more than a second). Those tracks yield 20 windows, not 21.

We keep the strict 480,000-sample window rather than zero-padding to force a 21st.
Every stored vector is then real audio: padding would produce a 21st vector that is
99.9% a duplicate of window 20 shifted by 1 s, plus 0.1% silence. The store is still
allocated at 21 rows/track, `n_windows` records the true count, and the unused row
stays zero-filled -- the design already absorbs this without change.

Consequence: **510,064 real vectors** rather than 525,000, and "21 windows per
track" in the spec is really "20 or 21".

### Final corpus results

| | |
|---|---|
| Tracks | 25,000 |
| Embedded | 24,980 (99.92%) |
| Failed (corrupt mp3) | 15 |
| Short (<10 s) | 5 |
| Real vectors | 510,064 |
| Store | 1,075,200,128 bytes |
| Window split | 14,513 tracks x20, 10,466 x21, 1 x18 |

The original music-checkpoint store passed a full-scale kill/restart experiment:
400 completed tracks reset and re-embedded across two SIGKILLs, converging to the
same store bytes (`664ef3c0...`). The replacement passes store integrity, GPU
determinism, and crash-window regression checks. Its vector SHA-256 is
`67bfc23b7f26fe97c0751192be478ebfea50464462c334ed44cbc9af7026d3da`.

### Design notes

- **Fixed manifest pre-pass.** Row assignment is `manifest_index * 21`, a pure
  function of sorted track ID, so it never depends on completion order, thread
  scheduling, or filesystem `readdir` order. The store is sized for all tracks up
  front; failed and short tracks keep a zero-filled slot and SQLite records why.
- **Short tracks** write `k < 21` real vectors and zero-fill the rest. A partial
  trailing window is dropped rather than padded — `repeatpad` would put an
  artifact vector into the index.
- **No `segments` table.** Its three columns are derivable (`offset_s = row_id -
  row_start`), and a second table that must stay consistent with the mmap would
  break the atomic commit above. Materialize later if Phase 1/2 profiling shows
  the range scan hurts.

## Original music checkpoint: historical measurements

**2026-09-09 diagnostic update:** these are measurements of the original
`larger_clap_music` store, not the replacement general checkpoint. Their earlier
interpretation as inherent CLAP limitations was incorrect; see
[the retrieval investigation](RETRIEVAL_DIAGNOSIS.md).

Measured on the original store (510,064 vectors). Re-evaluate geometric and
aggregation hypotheses on the replacement vectors before proceeding to HNSW.

### The space is strongly anisotropic

| Property | Value |
|---|---|
| Mean pairwise cosine (audio-audio) | **0.871** |
| Std of pairwise cosine | 0.177 |
| Norm of the mean vector | 0.923 (0 would be isotropic) |
| Top-1 principal component variance share | 45% |
| Participation ratio | **4.3 effective dims of 512** |

Vectors concentrate in nearly the same direction. A fresh embed reproduces this
geometry, which establishes reproducibility but does not validate the checkpoint.

### Mean-centering does not help (tested, rejected)

Centering collapses mean pairwise cosine from 0.87 to 0.045, which looks like a
dramatic fix, but retrieval quality is unchanged:

| | genre@10 | mean cos | std |
|---|---|---|---|
| chance | 17.2% | | |
| raw | **31.5%** | 0.864 | 0.177 |
| centered | 32.6% | 0.045 | 0.441 |

The measured improvement was small. Centering followed by normalization can
change cosine rankings; it cannot be dismissed as a translation that preserves
distances. This experiment does not validate the original checkpoint.

Chance is 17.2%, not 1/16: the corpus is skewed (28% Rock, 25% Electronic), so
that is the sum of squared genre shares. Reproduce with
`PYTHONPATH=src python3 tests/centering_test.py 2000`.

### Discrimination is real but weak

31.5% genre@10 against 17.2% chance is ~1.8x. Listening confirms it: neighbours
often share production era while differing in genre and instrumentation.

Earlier diagnostics considered these hypotheses, but did not establish causality:

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Hubness (universal neighbours) | top track appears 7x vs 0.4 expected; 71% of tracks appear 0 times | descriptive only |
| Mean-pooling washes out detail | within-track window similarity **0.979** | inconclusive given overlap and high unrelated-track cosine |
| Keying on encoding/production | genre agreement 37% > bitrate agreement 30.5% | inconclusive: different label distributions |

The controlled checkpoint comparison identifies the original checkpoint as the
leading cause: changing it raised genre@10 from 25.3% to 57.4% on the same
512-track candidate pool, with identical audio preprocessing.

### Earlier hypotheses for later phases (require new measurements)

- **Open question #1 now has a mechanism.** Standard HNSW benchmarks (SIFT, GIST)
  use far more isotropic data. A graph where nearly all distances sit near 0.87
  has much less structure to exploit, so published recall-vs-`efSearch` curves
  plausibly will not transfer. That is a defensible measured result.
- **Open question #2 matters more than it looks.** Within-track windows at 0.979
  mean a track's 20 windows are nearly one vector, so track-level mean-pooling
  discards most of the temporal resolution the 21x storage paid for. Max-pool and
  count-in-top-k should behave quite differently from mean.
- **The neighbour-selection heuristic risk is amplified.** Weak geometric signal
  gives less to distinguish good neighbours from bad, so a naive top-`M` graph
  will look fine and search badly. Phase 1 ground truth is the only thing that
  catches it.
