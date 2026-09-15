# Finish Timbre with GPT-5.3-Codex-Spark

Prepared 2026-09-11 for `/home/yohanb/timbre`. This is an execution guide for the
next agent, not a statement that the remaining work has been implemented.
Read `AGENTS.md` first. Current user instructions override this snapshot.

## Prompt to give Spark

> Read AGENTS.md and SPARK_HANDOFF.md. Finish Timbre's remaining core Phase 3
> work and connect the validated native index to the listening workflow. Work
> through the milestones in this guide, implementing and testing each before
> moving on. Run the real benchmarks and preserve their raw results. Continue
> through ordinary fixes and validation without asking permission at every
> milestone. Keep the existing corpus, embeddings, oracles and medium benchmarks
> intact. Do not begin the optional web demo or new product features. Keep me
> informed during long runs and honor any resource limits or pauses I request.
> Finish with reproducible results, updated documentation, and a precise account
> of anything still incomplete. Do not claim completion from code changes alone.

Select **GPT-5.3-Codex-Spark** in the model picker. The documented CLI model ID is:

```bash
codex -m gpt-5.3-codex-spark
```

OpenAI documents Spark as a text-only model for rapid coding iteration. The
milestone sizes, explicit tests, and written checkpoints below are this
project's recommended workflow, not claims about undocumented model behavior.
Use source, terminal output and JSON reports as evidence; do not require image
input to complete this project. Availability depends on the user's environment.
[Official model documentation](https://learn.chatgpt.com/docs/models).

## Working loop for this handoff

Use one small implementation milestone at a time. Before editing, name the
specific problem, files involved and check that will prove it fixed. Implement,
run the relevant checks, inspect the diff, then advance. Explicitly run tests;
do not leave a list of suggested tests as the deliverable.

Maintain a short progress entry in `NEXT_STEPS.md` after each milestone:

```text
Completed: implementation + artifact paths + measured checks
Running: exact command/service, log path, last completed checkpoint
Next: one concrete action
Open issues: failure evidence or unresolved measurement
```

Use that entry to recover after interruption. Inspect existing processes and
artifacts before restarting a job. Avoid simultaneous large builds, full-store
test runs and benchmarks: they compete for RAM and distort timings. Report
progress approximately once a minute while actively supervising long work.
When a check fails, diagnose and fix it; do not weaken the check to get a pass.

The historical 30–45 minute estimate covered the first large build and normal
benchmark only. It did not cover all implementation, memory experiments,
locality work, or this complete handoff. Replace estimates with observed rates.

## Snapshot: what exists

The project is an audio-similarity engine with a hand-written HNSW index.
Phases 0–2 are implemented. Phase 3's native C++ search and construction work on
125K and medium data. Large vectorization and exact ground truth are complete.
**No large graph or large index benchmark exists at this handoff.**

| Item | Verified state |
|---|---|
| Branch | `phase-3-native-search` |
| HEAD at handoff | `75dd4ed` — full-medium benchmark/tooling |
| Working tree | Substantial modified and untracked implementation; preserve it |
| Model | `laion/larger_clap_general` |
| Model revision | `ada0c23a36c4e8582805bb38fec3905903f18b41` |
| Medium corpus | 24,980 embedded tracks, 510,064 populated vectors |
| Large corpus | 106,574 processed: 105,884 embedded, 162 failed, 528 short, zero pending |
| Large vectors | 2,144,867 populated, dimension 512, float32 |
| Large allocation | 2,238,054 rows; unused rows are zero-filled holes |
| Large index candidates | 2,143,867 populated rows after holding out 1,000 queries |
| Hardware last observed | Ryzen 7 5800H, RTX 3060 6 GB, WSL2 with about 7.4 GiB RAM and 2 GiB swap |
| Environment | Python 3.10.12, NumPy 2.2.6, torch 2.13.0+cu130, transformers 4.53.0 |
| Native build | GCC 11.4.0, pybind11 3.1.0; extension already built locally |

Reinspect current resources and Git state; these values are a snapshot.
GPU inference is not needed to construct or query an index from stored audio
vectors. Text queries and embedding regression tests do need the model.

Both audio ZIPs were intentionally deleted. Their extracted MP3s remain.
Windows disk compaction reclaimed 110.60 GiB and its reminder was disabled.
There is no pending extraction, vectorization or disk-reclamation job to resume.

### Artifacts to reuse

| Path | Purpose |
|---|---|
| `store/timbre.db`, `store/vectors.npy` | Completed medium manifest/vectors |
| `store/groundtruth.npz` | Medium full-store oracle |
| `store/hnsw_phase2/`, `store/hnsw_medium/` | Original Python benchmark graphs/oracles/results |
| `store/hnsw_construction_125k/`, `store/hnsw_construction_medium/` | Paired Python/C++ construction artifacts, including `cpp.npz` |
| `store/large/timbre.db`, `store/large/vectors.npy` | Completed large manifest/vectors |
| `store/large/completion.json` | Ingestion counts and verified source hashes |
| `store/large/groundtruth.npz` | Full-store top-100, both self-excluded and whole-query-track-excluded variants |
| `store/large/index_groundtruth.npz` | **Use this to score the large held-out index** |
| `docs/large_groundtruth_results.json` | Oracle hashes, membership, timings, validation evidence |
| `docs/hnsw_*results.json` | Historical measurements to preserve and compare |

`store/` and `data/` are ignored by Git. A fresh clone will not contain these
artifacts. This handoff assumes the same workspace. Do not accidentally commit
audio, model weights, vector arrays or graph binaries. Keep compact reports in
`docs/` and reproducible code in source/scripts.

### Large-store identities

```text
Source vectors SHA-256:
cd69ad97883b5af7e546b7d127227e32f09cabde062e82850c6c0a6fac02d595
Full-store groundtruth.npz SHA-256:
9e3bbb54bfbe36740c59924904dd97a74da5a7b1824f99bb429dd0c19c2af407
Held-out index_groundtruth.npz SHA-256:
e90117be0ed08332c05cde37866dc781a18c02c2c29d0d31617f7f04f57807b9
```

The full-store cache uses four BLAS threads and reproduces byte-for-byte.
The held-out oracle uses one BLAS thread, seed `20260909`, and three independent
full-scan checks passed with a 1e-6 float32 score tolerance. The two caches are
not interchangeable. Source identity includes a manifest hash, not just the
vector hash. The completion report's database hash is a canonical bookkeeping
hash from `timbre.verify.hash_db`, not a raw hash of the SQLite file.

The 1,000 queries include 535 with missing genre labels, treated as an unlabeled
stratum. High geometric recall does not establish musical relevance. Overlapping
windows from the same track dominate many nearest-neighbor lists.

## First actions: inspect, then run focused checks

```bash
cd /home/yohanb/timbre
cat AGENTS.md
git status --short
git branch --show-current
git log -3 --oneline
free -h
df -h .
```

Read the current summary in `NEXT_STEPS.md`, remaining work in `PHASE3.md`, and
Phase 3's acceptance criteria in `TIMBRE_SPEC.md`. Use `FMA_LARGE.md` only for
corpus history/recovery. Old references to paused ingestion in historical logs
are not current instructions.

Run the relevant native/search baseline before changing it:

```bash
USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_hnsw.py tests/test_hnsw_benchmark.py \
  tests/test_native_hnsw.py tests/test_native_construction.py \
  tests/test_search.py tests/test_cache_identity.py
```

These commands exist now. Future large-runner or memory-experiment commands must
be implemented and checked before documenting them as runnable.

There are earlier records of 70 non-slow and three GPU tests passing. Those are
historical, not proof for your changed checkout. After the recent lazy-import
fix, ten focused search/cache tests passed. A broader non-slow run was manually
interrupted after 16 passes during expensive cache validation; it was not a
completed full-suite run and did not report a test assertion failure.

### Code map and known obstacles

| File | Read it for |
|---|---|
| `src/timbre/hnsw.py` | Python reference, vector validation, serialization and source fingerprint |
| `src/timbre/_hnsw_native.cpp` | Distance kernel, native graph construction/search, graph ownership |
| `src/timbre/native_hnsw.py` | `NativeHNSW`, `NativeHNSWBuilder`, save/load/freeze and resume |
| `src/timbre/benchmark_hnsw.py` | Candidate/oracle `prepare`, current `measure` and Python sweep |
| `src/timbre/benchmark_construction.py` | Paired build protocol and native checkpoint schedule |
| `src/timbre/benchmark_native.py` | Repeated paired query timings |
| `src/timbre/groundtruth.py` | Valid row layout, identity, sampling, exact search and recall |
| `src/timbre/search.py` | Exact distinct-track retrieval with winning passage offsets |
| `tests/listen.py` | Audio/text listening, currently exact search only |
| `tests/compare_search.py` | Existing comparison, hard-coded to medium/125K Python graph |

Important implementation facts, verified at handoff:

- `benchmark_construction` expects a `--baseline` directory containing
  `results.json` and `groundtruth.npz`. The prepared large oracle is a standalone
  cache. **Do not fabricate baseline results or run a huge Python build just to
  satisfy this interface.** Add an explicit validated-oracle route or a small
  dedicated large runner while retaining the medium path.
- The benchmark's `store[candidates]` materializes a roughly 4.1 GiB array.
  `np.array(store[candidates])` can copy that again. A source mmap alone does not
  make these operations memory bounded.
- `NativeHNSWBuilder.__init__` invokes `HNSW` for validation. The reference
  constructor computes whole-array finiteness/norm checks and creates Python
  adjacency containers. Profile and remove avoidable large temporary allocations
  on the native path without dropping validation.
- Native loading currently passes through `HNSW.load`, which expands adjacency
  into Python objects. Native `SearchIndex` also copies layer arrays into C++
  `std::vector`s. A new Python mmap loader alone will not make graph access mmap.
- `_vector_hash` currently hashes a contiguous buffer view; do not claim it
  calls `.tobytes()`. Contiguity conversion can still copy noncontiguous input.
  Every checkpoint fingerprints vectors, so checkpoint frequency affects time.
- `measure` currently requests ten hits. Recall@100 requires requesting at
  least 100 and using `efSearch >= 100`; it cannot be inferred from top-10 output.

## Milestone A — a runnable large native build

**Outcome:** construct the complete held-out graph with existing embeddings and
oracle, with reproducible checkpoints and bounded temporary allocations.

1. Accept the prepared oracle as an actual input. Validate its source identity,
   candidate/query arrays, dimensions, exclusions and seed before building.
   Preserve all existing medium commands and artifact formats where possible.
2. Plan vector memory explicitly. A simple option is a separate candidate-order
   float32 `.npy`, written in bounded chunks and reopened read-only. Its row order
   must equal the oracle's candidate order; external IDs remain original source
   row IDs. Record source identity and completion atomically. This is a copy of
   existing vectors, not new embedding inference.
3. Fix oversized validation temporaries if required. Keep the same finiteness,
   unit-norm, shape and ID checks, performed in bounded chunks. Do not change the
   distance kernel or relax floating-point semantics to work around memory use.
4. Begin with `M=8`, `efConstruction=80`, seed `20260909`, shuffled order from
   `default_rng(seed + 1).permutation(...)`, as used by the existing benchmarks.
5. Keep durable checkpoints. Resume with the same native backend, original
   insertion order and remaining nodes. Never label a resumed build's wall time
   as a fresh build time. Preserve or explicitly record any checkpoint-frequency
   difference from the previous 25,000-node schedule.
6. Use a supervised process with a persistent log and record the exact launch
   command. Estimate completion from insertion progress, not corpus vectorization
   rates. Honor subsequent pause/resource requests without modifying the corpus.

**Gate before the full run:** relevant existing tests pass; small synthetic or
existing benchmark data demonstrate input rejection, correct ID mapping,
checkpoint resume and bounded staging. Do not repeat a costly medium build if
the same behavior has already been verified and no change requires it.

**Gate after the run:** exactly 2,143,867 inserted nodes, saved graph identity
matches the candidate vector order, all query rows remain excluded, validation
passes, and a fresh process reloads the graph. Record initialization, insertion,
checkpoint and final-save times separately, plus peak memory and graph bytes.
If reload hits the known Python-expansion problem, implement the minimum direct
loader from Milestone C and return to this gate; do not rebuild a valid graph.

Suggested new artifact directory: `store/hnsw_large/`. This name is a proposal,
not an existing runner contract. Record whichever layout you implement.

## Milestone B — trustworthy large search measurements

**Outcome:** a reproducible large recall/latency curve and reload check.

- Query the same 1,000 held-out vectors against their matching candidate oracle.
  Start the top-10 sweep at `efSearch = 16, 32, 64, 128, 256`.
- Report recall@1 and recall@10, p50/p95/p99 latency, distance evaluations,
  reciprocal mean query time, graph size and memory. Separately measure
  recall@100 with valid beam sizes; do not alter the top-10 protocol silently.
- Pin one BLAS thread before NumPy import. Document warmup, query order and
  repeated passes. Follow the existing three-pass approach when comparing query
  timings; distinguish median per-pass percentiles from pooled percentiles.
- Time querying separately from loading, hashing, graph conversion, model
  inference and audio playback. Also record startup/load time for practical use.
- Require at least one normal operating point with recall@10 >= 0.95. Report low
  beam settings that fail; do not imply every setting passes. The existing
  construction harness uses an all-points gate, unlike the Phase 2 any-point
  gate. Resolve and document that policy rather than accidentally conflating it
  with the specification's minimum successful operating point.
- Reload in a fresh process and reproduce the selected setting's neighbors and
  recall. Use saved outputs/hashes as evidence, not just a successful load call.

Compare these historical measurements accurately:

| Candidates | C++ build, including saves | Recall@10 at efSearch=64 | C++ median / p95 |
|---|---:|---:|---|
| 125,000 | 36.99 s | 99.43% | 0.204 / 0.269 ms, query-only comparison |
| 509,064 | 170.76 s | 99.85% | 0.193 / 0.267 ms, query-only comparison |
| 2,143,867 | To measure | To measure | To measure |

The C++-constructed medium graph separately measured 0.187 / 0.265 ms. Do not
mix experiment definitions to manufacture a speedup. The large sample has
different queries/data; report scale trends without claiming identical workload.

**Gate:** raw JSON and a readable report contain the complete sweep, protocol,
source/oracle/graph hashes and successful reload evidence. Save compact report
copies in `docs/`. Large code existing is not equivalent to large measurements.

## Milestone C — direct loading and the memory experiment

**Outcome:** query a persistent graph without Python adjacency expansion, and
measure behavior when its vector working set exceeds available RAM.

Implement a simple versioned directory of mmap-friendly arrays if needed;
ordinary NPZ member loading is not a substitute for direct `.npy` mapping.
Persist vectors, external IDs, levels/layer arrays and metadata with explicit
fingerprints and completion rules. Preserve the old checkpoint reader or provide
a checked conversion path; do not require re-embedding or graph rebuilding.

In C++, retain owners for referenced read-only NumPy/mmap arrays. Validate sizes,
offset monotonicity and bounds, link targets, entry point and layer membership.
Never expose a dangling pointer when a Python temporary goes out of scope.
Test wrong fingerprints, malformed arrays, empty/partial graphs where supported,
owner lifetime and concurrent queries. A read-only query mapping must not be
mutated by a builder or by another session.

Compare baseline and direct loading on the same saved graph: identical graph
semantics and external result IDs, equivalent scores/recall, measured startup
and resident memory. Reuse the existing native tests and add focused regression
tests for the new lifetime and corruption risks.

Then run isolated query processes under cgroup limits. Suggested levels are
6 GiB, 4 GiB and 2 GiB, adjusted to actual machine availability. Set and record
swap policy explicitly; use zero allowed swap for a clean RAM-cap experiment.
Measure actual `memory.current`/`memory.peak`, anonymous/file memory, page faults,
query percentiles, recall and OOM/exit status. A shell variable or allocator
target alone is not an enforced memory cap.

Use the same graph, queries, beam and warmup protocol at every cap. Separate
startup from steady-state timing. Hash validation can warm the complete vector
file, so define cold versus warm procedures honestly. Account for file-cache
ownership/shared pages in cgroup measurements; a warmed cache outside the test
group can confound apparent limits. Avoid globally dropping caches or changing
WSL limits as a shortcut while the user is using the PC.

**Gate:** a repeatable table/plot shows actual resource limits and observed
latency/page-fault behavior. OOM is an observed outcome, not a latency number.
If the implementation cannot query under a cap, document why and fix avoidable
resident allocations before claiming to have measured graph traversal on disk.

## Milestone D — locality reordering

**Outcome:** compare the same graph and vectors before/after a physical layout
permutation, particularly under the memory caps.

Use a deterministic BFS-style graph traversal, visiting disconnected nodes in a
documented deterministic order. Save the permutation and inverse. Reorder vector
rows, levels and all graph node/link references consistently, including upper
layers and entry point. Preserve the mapping to original external source-row
IDs. Save new artifacts; keep the original graph and oracle intact.

Do not rebuild HNSW and call the difference a layout effect: rebuilding changes
the graph. Test the mapping as a bijection and verify edge/vector equivalence
after undoing the permutation. Existing beam ties use internal node IDs; their
order can change under permutation. Measure resulting neighbor/recall changes
and disclose tie effects rather than promising automatic bit-identical queries.

Repeat the same cap/query protocol and show whether locality helps, hurts, or
does nothing. A negative result is valid. **Gate:** correctness tests, mapping
metadata and comparable measured results all exist.

## Milestone E — usable native listening

The user's exact listening command worked but had a long delay. Unnecessary
PyTorch/Transformers imports were made lazy in `embed.py`, and `tests/listen.py`
now prints catalog/model/search status and elapsed search time. Preserve that
fix. Audio queries reuse stored embeddings; text queries load the pinned model.

Extend the existing listening/comparison workflow with an explicit native-index
option, retaining exact search as a reference. Inspect actual CLI parsers first:
`tests/compare_search.py` currently has hard-coded medium/125K paths.

Required behavior: random or selected audio passage and text queries return
distinct tracks, exclude the audio query's entire track, and play the winning
ten-second passage at its real offset. Convert graph external IDs through the
manifest; internal node numbers are not original row IDs after staging/reorder.

Overfetch segment neighbors and expand the beam/candidate count as necessary to
obtain the requested number of distinct tracks. Deduplicating only the first
five segment hits will often return one track. Be explicit about approximate
track ranking; segment recall does not establish exact distinct-track recall.
When comparing exact/ANN, use identical eligible candidates. A graph with held-out
queries does not have precisely the same membership as the complete corpus.

**Gate:** automated tests cover sibling exclusion, distinct tracks, winning
offsets, remapped IDs and insufficient candidate pools. Run an audio and a text
smoke query. Provide the user with commands that actually use the saved native
index. Only human feedback can establish that the matches sound good.

## Finish the documented experiments, then close out

After A–E, finish the remaining benchmark items already listed in `PHASE3.md`:
recall@100, multithread throughput and hnswlib/FAISS comparisons. Use separate
optional benchmark dependencies and official library docs when implementing
them. The project's own index must remain hand-written; these libraries are
comparators, not replacements. Match candidates, queries, distance metric,
exclusions, thread counts and timing scope. Compare at similar recall as well as
similar parameters. Record versions and include full memory costs for each.

Multithread throughput must use a shared read-only index, validated result
correctness, warmup and explicit worker counts. Native search releases the GIL,
but wrapper bookkeeping such as `distance_evaluations` is mutable: avoid racing
that field when collecting concurrent metrics.

The optional Phase 4 web demo, new features, alternate embeddings, aggregation
research and product quantization are outside the default finish. Do not let
these expand the task. If the user later requests the demo, follow Phase 4 in
`TIMBRE_SPEC.md`: a minimal upload/text query page with five audio results.

### Final tests and build commands

Rebuild the native extension only when needed. The existing local header cache
is documented because this host lacked system Python development headers:

```bash
CFLAGS="-I$HOME/.cache/timbre/python-dev/usr/include/python3.10 -I$HOME/.cache/timbre/python-dev/usr/include" \
CXXFLAGS="-I$HOME/.cache/timbre/python-dev/usr/include/python3.10 -I$HOME/.cache/timbre/python-dev/usr/include" \
  python3 -m pip install --no-deps -e .
```

Preserve C++17, `-O3 -ffp-contract=off`, and the pinned dependency versions unless
a demonstrated requirement calls for a documented change. The built `.so` is
ignored; a fresh checkout needs its own build.

Run focused tests during development, then the complete relevant suite before
closing out. Run the GPU regression tests too if embedding/import changes remain
in the final changeset (they do at this handoff):

```bash
USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest -q -m 'not slow'
USE_TF=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest -q -m slow
git diff --check
```

The non-slow suite includes real medium-store ground-truth rebuilds, so it is
not a subsecond smoke suite. Run it separately from performance measurements.
Never run `tests/kill_restart.sh` against the completed production store merely
to demonstrate recovery; use isolated fixtures/checkpoints for destructive tests.

Update `README.md`, `PHASE3.md`, `NEXT_STEPS.md` and relevant `AGENTS.md` status
sections with what actually passed. Preserve unrelated persistent instructions.
Include build/query/recall/memory results from 125K through the large corpus,
before/after locality plots, limits of the conclusions, and exact reproduction
commands. Store plotting code and shareable plots; do not hand-edit chart values.

Review all modified and untracked files before a commit: much of the native
implementation predates this handoff and is not in HEAD. Do not reset or discard
it. Follow the user's current commit/push instruction; there is no Timbre VPS
deployment task here. A clean clone alone will not reproduce results without
the ignored data artifacts, so document the data prerequisites.

### Definition of finished

- [ ] Large native graph built, checkpointed, reloaded and matched to its oracle.
- [ ] Measured large recall/latency curve and complete build/memory report.
- [ ] Direct loading verified without Python adjacency expansion or hidden graph copies.
- [ ] Enforced memory-cap experiment measured, including page faults and failures.
- [ ] Locality permutation validated and compared under the same protocol.
- [ ] Native listening commands work and preserve passage/track semantics.
- [ ] Recall@100, multithread and library-comparison items have measured results.
- [ ] Applicable tests pass; failures or unrun checks are explicitly reported.
- [ ] Reports, plots, reproduction instructions and project status agree.
- [ ] Optional demo/features are clearly deferred rather than silently claimed complete.

If any required box is open, report partial completion with the exact remaining
action. When all are closed, state the measured result and remaining optional
work plainly. Do not create an endless new optimization backlog.
