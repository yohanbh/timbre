# Phase 3 — native search and construction

The native search and construction milestones are complete on 125K and full-medium
data. Hand-written C++ distance, search, insertion, neighbor selection and pruning
are exposed through pybind11. Query-only comparisons use the same saved Python
graphs; construction comparisons build separate Python/C++ graphs against the same
validated candidates and exact oracles. Both reuse the existing embeddings.

Phase 3 as a whole is **not complete**. The user authorized FMA large expansion
on 2026-09-10. Download, extraction and vectorization are complete. All 106,574
tracks were processed: 105,884 embedded, 162 failed and 528 too short, with zero
pending and 2,144,867 populated vectors. Final store verification passed.
See [FMA_LARGE.md](FMA_LARGE.md).
The large-corpus measurements, memory caps and locality reordering remain.

## Implementation

`src/timbre/_hnsw_native.cpp` traverses compact adjacency arrays, using a nearest-first
candidate heap, a bounded farthest-first result heap and per-query visited state.
It preserves Python's internal-node tie breaking during beam search and external-ID
tie breaking when ranking results. No third-party ANN implementation is used.

`NativeHNSW` in `src/timbre/native_hnsw.py` freezes a Python graph for search.
`NativeHNSW.load(path, vectors)` uses the existing graph loader to verify the exact
vector fingerprint and graph structure, then converts the adjacency representation.
Consequently, loading still temporarily materializes Python adjacency lists. It is
not yet a direct memory-mapped graph loader.

Query normalization stays in Python and matches `HNSW.search`. The boundary is
crossed once per query; C++ performs all graph traversal and dot products while
releasing the GIL. Vector memory is shared when C-contiguous and must remain
immutable. Graph links and IDs are copied, so later insertion into the Python
index does not change an existing native snapshot.

The float32 dot product uses eight independent accumulators. The build uses
`-O3 -ffp-contract=off`, C++17, without fast-math or architecture-specific flags.
Its reduction order differs from NumPy/BLAS, so bit-identical scores and traversal
are not promised. See the measured neighbor agreement below.
Bindings and packaging follow the official pybind11
[NumPy interface](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
and [build helpers](https://pybind11.readthedocs.io/en/stable/compiling.html).

## Native graph construction

`NativeHNSWBuilder` adds mutable C++ adjacency, insertion, reciprocal links,
diversified neighbor selection and overflow pruning. Construction and immutable
query snapshots share the same C++ distance, greedy-descent and beam-search code.
The original Python implementation remains available as the reference.

Python validates vectors and IDs and assigns exponential levels using the same
NumPy generator and formula as `HNSW`. It sends up to 1,000 nodes per native batch.
The C++ builder is serial and holds the GIL during each batch; immutable query
snapshots still release the GIL during search. `build(order, progress)` invokes
the callback after each batch, allowing checkpointing between completed batches.
`freeze()` copies adjacency into a search snapshot that later insertion cannot
change.

`save()` writes the existing NPZ graph format atomically, including RNG state,
the exact vector fingerprint and a construction-backend identifier. Both Python
`HNSW.load` and `NativeHNSW.load` can query these graphs. To resume construction,
use `NativeHNSWBuilder.load`, then supply the **remaining original insertion order**.
Calling `build()` without an order inserts remaining rows in row order. Native
resume rejects checkpoints from other construction backends; low-bit differences
in distance computations can change graph edges, so switching backends is not a
promise of identical continuation. Determinism assumes the same software,
hardware and native kernel. Loading currently uses the validated Python loader and temporary
Python adjacency, rather than a memory-mapped graph representation.

Reproduce paired construction measurements using the existing medium embeddings:

```bash
PYTHONPATH=src python3 -m timbre.benchmark_construction \
  --baseline store/hnsw_phase2 --out store/hnsw_construction_125k

PYTHONPATH=src python3 -m timbre.benchmark_construction \
  --baseline store/hnsw_medium --out store/hnsw_construction_medium
```

The harness builds Python and C++ graphs with identical candidate membership,
insertion order, seed, `M=8` and `efConstruction=80`. It checks that every node
received the same level. It reuses the validated exact oracle and searches both
graphs with C++ at efSearch 16, 32, 64, 128 and 256. Each graph is saved every
25,000 insertions and once after completion. Reports separate initialization,
insertion, checkpoint writing and final saving; total build time includes all
four and excludes oracle loading and conversion to a query snapshot. These are
single fresh builds, not repeated-build medians. A resumed run cannot claim a
fresh total build time. Rerunning an already completed output reuses its saved
graph/build timing; use a new output directory for new construction measurements.

### Construction measurements — 2026-09-10

| Candidate segments | Python total build | C++ total build | Total speedup | Python insertion | C++ insertion |
|---|---|---|---|---|---|
| 125,000 | 195.44 s | 36.99 s | 5.28× | 190.71 s | 35.17 s |
| 509,064 | 826.95 s | 170.76 s | 4.84× | 769.99 s | 151.29 s |

At 125K, insertion alone improved **5.42×**. Recall@10 was unchanged at all five
settings: 95.72%, 98.36%, 99.43%, 99.81% and 99.94% respectively. The fresh Python
graph is byte-identical to the original Phase 2 graph, and the Python/C++ level
arrays have identical SHA-256 hashes. The C++ graph is a separate artifact;
matching layer assignments do not imply identical adjacency across backends.
[Full 125K construction results](docs/hnsw_construction_125k_results.json).

On full medium, insertion alone improved **5.09×**. Total construction fell from
**13 minutes 47 seconds to 2 minutes 51 seconds**. Recall@10 was again unchanged
at every setting: 98.92%, 99.60%, 99.85%, 99.89% and 99.90%. The Python graph was
byte-identical to the earlier full-medium graph; all Python/C++ node levels match.
At efSearch=64, the C++-built graph measured 0.187 ms median and 0.265 ms p95 using
native query search. Both graphs are queried in C++ for this comparison.
[Full-medium construction results](docs/hnsw_construction_medium_results.json).

Both native checkpoints were reloaded through `NativeHNSWBuilder.load`, validated
and frozen for search. Running all 1,000 queries again reproduced recall@10 of
99.43% and 99.85% at efSearch=64. Native serialized graph sizes are 8.89 MiB and
34.89 MiB, excluding vectors. The complete comparison processes peaked at
2.16 GiB and 2.93 GiB respectively; these include the sequential Python and C++
builds, validation, saving and query conversion, not native-only construction RSS.

## Paired query measurements — 2026-09-10

Both sizes use `M=8`, `efConstruction=80`, shuffled insertion, seed `20260909`,
and the fixed 1,000 held-out query rows. Each exact oracle covers precisely that
graph's candidate set. Source cache identity, candidate membership, graph IDs,
configuration and completeness are checked before timing.

Each setting runs three passes per backend, alternating which backend goes first.
A pass warms ten queries, then times all 1,000 queries individually, including
Python normalization and result conversion. Reported latency percentiles are the
**median of the three per-pass percentiles**, not percentiles of pooled samples.
Loading, graph conversion and model inference are excluded. JSON reports retain
every pass, p99, reciprocal mean query latency, recall@1, distance evaluations,
graph/oracle hashes, compiler and library versions.

Hardware: AMD Ryzen 7 5800H, 7.4 GiB RAM, WSL2 Linux, one CPU/BLAS thread,
warm caches. Python 3.10.12, NumPy 2.2.6, GCC 11.4.0, pybind11 3.1.0.
The Python timings below were freshly measured alongside C++; historical Phase 2
timings remain in their original reports and are not the speedup denominator.

### 125,000 candidate segments

| efSearch | Recall@10, both | Python median | C++ median | Python p95 | C++ p95 | Median speedup |
|---|---|---|---|---|---|---|
| 16 | 95.72% | 0.371 ms | 0.078 ms | 0.595 ms | 0.117 ms | 4.76× |
| 32 | 98.36% | 0.536 ms | 0.122 ms | 0.681 ms | 0.173 ms | 4.40× |
| 64 | 99.43% | 0.886 ms | 0.204 ms | 1.060 ms | 0.269 ms | 4.35× |
| 128 | 99.81% | 1.585 ms | 0.364 ms | 1.840 ms | 0.484 ms | 4.35× |
| 256 | 99.94% | 2.952 ms | 0.654 ms | 3.404 ms | 0.842 ms | 4.51× |

All top-10 neighbor sets and their order matched Python for all 1,000 queries at
all five settings. Native graph conversion took 0.50 seconds.
[Full results](docs/hnsw_native_125k_results.json).

### Full medium: 509,064 candidate segments

| efSearch | Recall@10, both | Python median | C++ median | Python p95 | C++ p95 | Median speedup |
|---|---|---|---|---|---|---|
| 16 | 98.92% | 0.363 ms | 0.075 ms | 0.520 ms | 0.118 ms | 4.82× |
| 32 | 99.60% | 0.535 ms | 0.116 ms | 0.729 ms | 0.167 ms | 4.62× |
| 64 | 99.85% | 0.879 ms | 0.193 ms | 1.091 ms | 0.267 ms | 4.55× |
| 128 | 99.89% | 1.529 ms | 0.346 ms | 1.809 ms | 0.452 ms | 4.41× |
| 256 | 99.90% | 2.903 ms | 0.632 ms | 3.402 ms | 0.821 ms | 4.59× |

All top-10 neighbor sets matched Python for all 1,000 queries at every setting.
Exact result ordering matched for 997/1,000 queries at each setting. Recall@1
and recall@10 were unchanged. Native conversion took 2.02 seconds.
[Full results](docs/hnsw_native_medium_results.json).

At efSearch=64, inspecting those three queries confirmed near-tied scores became
exact float32 ties in C++. Query rows `169900`, `382272` and `453376` swapped ranks
6/7, 2/3 and 4/5 respectively. The affected scores differed between backends by
at most `1.19e-7`; the external-ID tie breaker then changed their order.

The comparison processes peaked at 2.16 GiB (125K) and 2.89 GiB (medium). These
figures include loading, validation and **both** backends. They do not measure
native-only serving memory or behavior under a memory cap. Graph build times
in this earlier query-only experiment were the Python measurements: 189.1 seconds
and 862.7 seconds respectively. The construction comparison above separately
measures new builds of both backends with a shared timing protocol.

Medium query latency need not exceed 125K latency: the graphs and exact oracles
differ, and medium search visits fewer vectors at these settings. Neither result
predicts latency at 2.2M vectors or beyond RAM capacity. Moreover, 94.93% of the
medium oracle's exact top-10 hits are same-track windows. Unchanged geometric
recall establishes index behavior, not musical relevance.

## Build and reproduce

A C++17 compiler and development headers for the active Python interpreter are
required. The build installs pybind11 through `pyproject.toml`.

```bash
python3 -m pip install --no-deps -e .

PYTHONPATH=src python3 -m timbre.benchmark_native \
  --baseline store/hnsw_phase2 --out store/hnsw_native_125k/results.json

PYTHONPATH=src python3 -m timbre.benchmark_native \
  --baseline store/hnsw_medium --out store/hnsw_native_medium/results.json

USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python3 -m pytest -q -m 'not slow'
```

The query-only `benchmark_native` requires completed Phase 2 artifacts and reuses
their exact oracles. It does not rebuild graphs or embeddings and rejects
overwriting baseline files. The construction benchmark above saves new graphs
in separate output directories.
The extension is already built in this workspace. Shared libraries and build
outputs are ignored by Git; another checkout must build its own extension.

Validation passed: **68 non-slow tests**, with three GPU tests deselected. Native
tests cover full-beam exact neighbors, Python agreement across dimensions and
beam widths, empty/partial graphs, tied vectors and external IDs, snapshot
isolation, fingerprint rejection, noncontiguous inputs, owner lifetime,
concurrent queries and malformed graph arrays. Construction tests additionally
cover small-graph parity with Python, directional neighbor selection, clustered
held-out recall, identical graphs/RNG state after multiple checkpoint resumes,
and recovery after discarding 1,000 uncheckpointed insertions. Query and
construction report copies and graph/oracle hashes were verified; the original
reports are unchanged.

This host lacked system Python headers. Matching Ubuntu `libpython3.10-dev`
headers were unpacked without root into `~/.cache/timbre/python-dev`. To rebuild
using that existing cache:

```bash
CFLAGS="-I$HOME/.cache/timbre/python-dev/usr/include/python3.10 -I$HOME/.cache/timbre/python-dev/usr/include" \
CXXFLAGS="-I$HOME/.cache/timbre/python-dev/usr/include/python3.10 -I$HOME/.cache/timbre/python-dev/usr/include" \
  python3 -m pip install --no-deps -e .
```

## Large-store ground truth — 2026-09-11

Ground-truth preparation is complete; no large HNSW graph has been built yet.

| Artifact | Candidates | Queries | Exclusions | BLAS threads |
|---|---:|---:|---|---:|
| `store/large/groundtruth.npz` | 2,144,867 populated rows | 1,000 | Query itself; separate variant excludes its whole track | 4 |
| `store/large/index_groundtruth.npz` | 2,143,867 populated rows | Same 1,000 | Every query row excluded from candidates | 1 |

Both contain exact top-100 answers, use seed `20260909`, and record source
identity. Use the second cache when evaluating the held-out large graph.
It uses the existing `benchmark_hnsw.prepare` format; graph construction has
not been run merely to create a baseline directory or results file.

The full-store passes took 37.3 s and 27.4 s. A fresh rebuild took 31.6 s and
32.1 s and produced a byte-identical NPZ. Held-out oracle preparation took
59.05 s. These are computation timings, excluding the full preparation and
validation workflow. All query IDs, populated candidate membership, unique
results, exclusions, finite ordered scores and source hashes passed validation.
Independent per-query scans over every candidate checked three held-out queries,
allowing 1e-6 score differences between GEMV/GEMM and near-tied boundaries.

Sampling follows the existing genre-stratified policy, including an unlabeled
stratum: 535 of the 1,000 queries have no genre label. This cache evaluates
geometric retrieval; it does not establish musical relevance.

[Validation details, hashes and timings](docs/large_groundtruth_results.json).

## Remaining Phase 3 work

1. Build and benchmark the large C++ index using the completed held-out oracle
   below. Preserve the completed medium artifacts; no re-vectorization is needed.
   Suggested commands:

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

   # Recall@100: run a separate sweep with k=100 and valid efSearch values.
   PYTHONPATH=src python3 -m timbre.benchmark_native \
     --baseline store/hnsw_medium --out store/hnsw_native_large_recall100/results.json \
     --oracle store/large/index_groundtruth.npz --graph store/hnsw_construction_large/cpp_direct \
     --db store/large/timbre.db --store store/large/vectors.npy \
     --groundtruth store/large/groundtruth.npz --m 8 --ef-construction 80 \
     --ef-search 128 256 512 --k 100
   ```
2. Load vectors and graph arrays without Python adjacency expansion. Measure query
   RSS, page faults and tail latency under a hard memory cap, then repeat with nodes
   and vectors reordered by graph locality.
3. Report build time, query latency and recall from 125K through the actual large
   corpus size. Add recall@100, multithread throughput and hnswlib/FAISS baselines.

The optional demo and musical-relevance experiments remain separate work.
