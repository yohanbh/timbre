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

The **full-medium baseline** is also complete: **509,064 indexed segments** plus
the same 1,000 held-out queries, using the existing embeddings. At
`M=8, efConstruction=80, efSearch=64`, it reaches **99.85% recall@10, 1.139 ms
median and 1.835 ms p95**. Build time was **14 minutes 23 seconds**; the serialized
graph is **34.89 MiB**, excluding vectors. Reloading reproduces the recall result.
This is one graph with five search settings. See the
[full-medium protocol and comparison](PHASE2.md#full-medium-baseline-2026-09-10)
and [measured results](docs/hnsw_medium_results.json). Most exact neighbors at
this size are overlapping windows from the query's own track; musical relevance
requires separate evaluation.

## Phase 3 — native search and construction

The hand-written C++ distance and search loop is implemented behind pybind11.
On the same saved graphs and 1,000 held-out queries, at `efSearch=64`:

| Indexed segments | Python median | C++ median | Speedup | Recall@10, both |
|---|---|---|---|---|
| 125,000 | 0.886 ms | 0.204 ms | 4.35× | 99.43% |
| 509,064 | 0.879 ms | 0.193 ms | 4.55× | 99.85% |

These are fresh paired measurements on one CPU thread, with three passes per
backend. Every top-10 neighbor set matched across all five tested efSearch
settings; three medium-store queries differed in result order. Native medium
p95 at `efSearch=64` was **0.267 ms**. That query-only experiment reused saved graphs.

C++ graph construction is also implemented, including neighbor selection,
reciprocal insertion, pruning and deterministic checkpoint continuation. Fresh
builds with identical order, node levels and checkpoint frequency measured:

| Indexed segments | Python build | C++ build | Speedup | Recall@10 at efSearch=64, both |
|---|---|---|---|---|
| 125,000 | 195.44 s | 36.99 s | 5.28× | 99.43% |
| 509,064 | 826.95 s | 170.76 s | 4.84× | 99.85% |

Total build time includes initialization and checkpoint/final saves. Recall was
unchanged at all five tested search settings; reloading both native graphs
reproduced the efSearch=64 results. These are single fresh builds per backend.

See [implementation, setup and full measurements](PHASE3.md).
Phase 3 remains in progress. FMA large download, extraction, vectorization and
final verification completed on 2026-09-10. The large store contains 2,144,867
populated vectors from 105,884 embedded tracks; 162 failed and 528 were too short,
with zero pending. No re-vectorization is needed for the index benchmarks.
See [the large-corpus run and progress commands](FMA_LARGE.md).

The large native graph is built and **passes the recall gate on all 2,143,867
indexed segments**, with 1,000 held-out queries at `M=8, efConstruction=80`:

| efSearch | recall@10 | mean distance evals | p50 | p95 |
|---|---|---|---|---|
| 16 | 97.25% | 245 | 0.081 ms | 0.178 ms |
| 32 | 99.02% | 348 | 0.080 ms | 0.145 ms |
| 64 | **99.58%** | 547 | **0.145 ms** | 0.249 ms |
| 128 | 99.85% | 933 | 0.396 ms | 0.880 ms |
| 256 | 99.92% | 1664 | 1.179 ms | 2.562 ms |

Recall holds across a 4.2x corpus increase: 99.85% on full-medium versus 99.58%
here, both at efSearch=64 — at 0.145 ms median against full-medium's 0.193 ms.

The graph is mapped with `NativeHNSW.load_directory`, which costs **0.097 GiB
RSS and 0.37 seconds** for 4,537,964,181 bytes on disk; arrays stay memory-mapped
and no Python adjacency is expanded. Latencies are measured with efSearch
settings interleaved per query against a warm page cache, so no setting absorbs
another's paging cost; recall and distance evaluations match the build-time
sweep exactly. An earlier ascending single-pass sweep against a cold cache
produced non-monotonic timings and is recorded as an artifact, not a result.

Page-cache state dominates latency and leaves recall untouched. At efSearch=64,
a cold start costs **171x at p99** (173.749 ms versus 1.015 ms, with 1,022 major
faults versus zero) at an unchanged 99.58% recall@10. Two negative results: a
`RLIMIT_AS` cap cannot express this workload, since address space must cover the
whole 4.09 GiB mapping and caps at or below 4 GiB fail outright; and
`madvise(MADV_DONTNEED)` does not emulate cold storage, returning every page
from page cache with zero major faults.

See [memory and loading measurements](docs/hnsw_memory_large_results.json) and
[the warm latency curve](docs/hnsw_warm_latency_large_results.json). Locality
reordering, multithreaded search and library baselines are still ahead.

## Open questions — measured results

### Insertion order does not damage the graph

FMA is ordered by track ID, which correlates with album and genre, so a naive
build inserts thousands of similar vectors consecutively. Measured on the real
corpus, **99.0%** of consecutive inserts in track order share the previous
track's genre, against **17.4%** shuffled — the clustering is real, 5.7x.

It changes nothing. Both graphs on 125,000 vectors at `M=8, efConstruction=80`,
with the same 1,000 held-out queries and the same exact oracle:

| efSearch | Shuffled | Track order | Delta |
|---|---|---|---|
| 16 | 95.72% | 95.64% | -0.08 |
| 32 | 98.36% | 98.34% | -0.02 |
| 64 | 99.43% | 99.41% | -0.02 |
| 128 | 99.81% | 99.74% | -0.07 |
| 256 | 99.94% | 99.95% | +0.01 |

Every difference is under 0.1 points and the sign is inconsistent, which is
noise rather than degradation. Track order did cost **13% more build time**
(212.89 s versus 189.15 s) and a marginally larger graph (9,453,842 versus
9,321,766 bytes), consistent with more pruning work when consecutive inserts
compete for the same neighbourhoods. The diversified neighbour-selection
heuristic appears to be what absorbs the clustering. Shuffled remains the
default. [Full results](docs/hnsw_order_track_results.json).

### Segment aggregation: genre agreement cannot separate the rules

A track has 20-21 chances to match, so ranking needs a rule to collapse window
hits into one track score. Compared on 2,000 queries over the medium store,
scored by genre agreement among the top 10 distinct tracks (chance 17.2%):

| Rule | genre@10 | SEM | ms/query |
|---|---|---|---|
| max — best window, current search behaviour | 67.0% | 0.8 | 1.41 |
| mean — average over real windows | 67.2% | 0.8 | 1.69 |
| topk — mean of the best 3 | 67.0% | 0.8 | 7.84 |
| count — windows above 0.8 | 67.0% | 0.8 | 3.00 |

The spread is **0.20 points against a SEM of 1.11**: indistinguishable.

That is not the same as the choice being irrelevant. The rules return
substantially different music — **max and mean share only 6.1 of 10 results and
produced an identical top-10 in 0 of 150 queries**. They disagree constantly and
score the same, which locates the limit of genre agreement as a relevance proxy
rather than settling the question. A listening comparison on the cases where the
rules disagree is recorded as open in [LISTENING.md](LISTENING.md). `max` stays
the default: it is the cheapest of the four and already what `exact_track_topk`
implements. [Full results](docs/aggregation_results.json).

Reproduce:

```bash
USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 tests/aggregation_test.py 2000
PYTHONPATH=src python3 -m timbre.benchmark_hnsw --vectors 125000 --queries 1000 \
  --m 8 --ef-construction 80 --ef-search 16 32 64 128 256 \
  --order track --out store/hnsw_order_track
```

### Locality reordering recovers nothing at 512-d float32

HNSW search is a pointer chase, so once the graph outgrows RAM every hop can
fault. `src/timbre/reorder.py` permutes node IDs into BFS order from the
entrypoint, so nodes reached together in a traversal sit together on disk.

The permutation is correct: on all **2,143,867** nodes it produced **identical
top-10 sets** at every efSearch setting, not merely equal recall. The tool fails
rather than reports if recall moves.

It does not help. Vector pages touched per layer-0 neighbourhood:

| Graph | Before | After |
|---|---|---|
| 125,000 | 10.09 | 10.12 |
| 2,143,867 | 9.73 | 9.84 |

The reason is arithmetic. A 512-dimensional float32 vector is **2,048 bytes —
exactly half a 4 KiB page**, so two vectors share a page and co-locating
neighbours requires landing them within about two consecutive IDs. With ~10.5
neighbours per node the floor is ~5.75 pages and both layouts sit near 9.8. BFS
did improve ID locality by 35% (mean neighbour-ID distance 26,634 to 17,324);
that improvement cannot cash out at this page-to-record ratio. BFS is in fact
*worse* on the measure that matters — 1.06% of neighbours within two IDs versus
16.56% for insertion order — because it numbers a node's neighbours across an
expanding frontier rather than adjacent to their source.

This is a result about 512-d float32 on 4 KiB pages, not a claim that no
reordering can ever help. A narrower vector, a larger page, or a layout that
co-locates a node *with its own neighbours* rather than with its BFS cohort
would each change the arithmetic.

**Cold-cache timings on this host are not reliable enough to distinguish the
layouts.** A first run showed BFS ahead at 1.28x on cold p99; a swapped-order
repeat inverted it. In both runs whichever layout was measured *second* won,
because `posix_fadvise` evicts one graph's files while the machine's broader
cache state carries over. The same layout varied up to **6x** in cold p99
between runs (885.4 ms versus 146.8 ms). The tell was that the apparently faster
layout also took *more* major faults, which is incoherent if locality were the
cause. Warm p99 is stable at **0.66-0.76 ms** for both layouts across every run.
A trustworthy cold comparison needs randomized order and repeats, not one pass
per layout. See [the results](docs/hnsw_locality_cold_results.json) and
[the swapped-order control](docs/hnsw_locality_cold_swapped_results.json).

Reproduce:

```bash
PYTHONPATH=src python3 -m timbre.reorder \
  --graph store/hnsw_construction_large/cpp_direct --out store/hnsw_locality_large \
  --oracle store/large/index_groundtruth.npz --store store/large/vectors.npy

USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 tests/locality_cold.py \
  --graphs store/hnsw_construction_large/cpp_direct store/hnsw_locality_large \
  --labels insertion bfs \
  --oracle store/large/index_groundtruth.npz --store store/large/vectors.npy
```

### Baseline gap: 1.5-2.0x behind hnswlib, and where it goes

hnswlib and FAISS are reference baselines only and never appear in the serving
path. All three built fresh in one process over the same **509,064** candidates
and 1,000 held-out queries, at `M=8, efConstruction=80`, one thread everywhere
(`OMP_NUM_THREADS=1`, `faiss.omp_set_num_threads(1)`, `hnswlib num_threads=1`),
one query per call rather than a batch, scored against the same exact oracle.

| efSearch | ours p50 | hnswlib p50 | FAISS p50 | gap vs hnswlib |
|---|---|---|---|---|
| 16 | 0.111 ms | 0.076 ms | 0.078 ms | 1.45x |
| 32 | 0.143 ms | 0.091 ms | 0.071 ms | 1.58x |
| 64 | 0.313 ms | 0.159 ms | 0.121 ms | 1.97x |
| 128 | 0.521 ms | 0.336 ms | 0.222 ms | 1.55x |
| 256 | 0.882 ms | 0.597 ms | 0.415 ms | 1.48x |

Recall is not what we are giving up. **Ours is the highest at every setting:**

| efSearch | ours | hnswlib | FAISS |
|---|---|---|---|
| 16 | **98.92%** | 97.76% | 98.22% |
| 32 | **99.60%** | 99.37% | 99.40% |
| 64 | **99.85%** | 99.74% | 99.75% |
| 128 | 99.89% | 99.83% | 99.88% |
| 256 | 99.90% | 99.84% | **99.92%** |

Build time: ours 163.5 s, FAISS 116.8 s, hnswlib 59.1 s.

**Where the time goes.** At `efSearch=64` a query costs 254.5 us over 475.5
distance evaluations — **536 ns per evaluation**. A 512-dimensional float32 dot
product is 512 multiply-adds; at ~3 GHz that is ~171 ns scalar, ~43 ns with
4-wide SSE2, and ~21 ns with 8-wide AVX2. We are an order of magnitude above
even the scalar estimate.

The kernel in `src/timbre/_hnsw_native.cpp` is written *for* vectorization —
eight independent accumulators specifically so a compiler can use SIMD without
fast-math reassociation — but it is compiled with `-O3 -ffp-contract=off` and no
architecture flag. `-O3` alone targets baseline x86-64, so it emits SSE2 and
never uses the AVX2 and FMA this CPU reports in `/proc/cpuinfo`. hnswlib ships
hand-written AVX/AVX-512 intrinsics with runtime dispatch. That single
difference is consistent with the entire measured gap, and the fix is a compile
flag rather than an algorithm change — deliberately not applied here, because
`-march=native` would break the bit-identical-output guarantee that Phase 0
established and every determinism test depends on.

Two caveats. This is the medium store, not large: three indexes at 2.14M vectors
need roughly 12 GiB against 7.4 GiB of RAM, and each library copies the vectors
internally. And these are single fresh builds, not repeated-build medians.
[Full results](docs/baseline_comparison_results.json).

```bash
python3 -m pip install --user hnswlib faiss-cpu   # or: pip install -e '.[baselines]'
USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 tests/baseline_compare.py \
  --graph store/hnsw_construction_large/cpp_direct \
  --oracle store/hnsw_medium/groundtruth.npz --store store/vectors.npy
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
| Mean-pooling washes out detail | within-track window similarity **0.979** | superseded: 0.917 on the replacement checkpoint, and temporal probes diverge — see [LISTENING.md](LISTENING.md) |
| Keying on encoding/production | genre agreement 37% > bitrate agreement 30.5% | retired: the spoken-word listening control found no production-over-content ranking — see [LISTENING.md](LISTENING.md) |

The controlled checkpoint comparison identifies the original checkpoint as the
leading cause: changing it raised genre@10 from 25.3% to 57.4% on the same
512-track candidate pool, with identical audio preprocessing.

### Earlier hypotheses for later phases (require new measurements)

- **Open question #1 now has a mechanism.** Standard HNSW benchmarks (SIFT, GIST)
  use far more isotropic data. A graph where nearly all distances sit near 0.87
  has much less structure to exploit, so published recall-vs-`efSearch` curves
  plausibly will not transfer. That is a defensible measured result.
- **Open question #2 matters more than it looks — and is now answered.** Within-track
  windows at 0.979 meant a track's 20 windows were nearly one vector on the original
  checkpoint. **This does not hold on the replacement checkpoint:** measured over 300
  sampled tracks, mean within-track pairwise cosine is **0.917**, and first-window
  versus last-window is **0.821**. Querying offset 0 against a late window shares a
  median of only **2/10** neighbours, with 11 of 40 sampled tracks fully disjoint, and
  listening confirms each result set suits its own passage. Temporal resolution is
  real, so the 21x storage earns its keep and max-pool versus count-in-top-k versus
  mean is a live experiment rather than a formality. See
  [the listening findings](LISTENING.md).
- **The neighbour-selection heuristic risk is amplified.** Weak geometric signal
  gives less to distinguish good neighbours from bad, so a naive top-`M` graph
  will look fine and search badly. Phase 1 ground truth is the only thing that
  catches it.
