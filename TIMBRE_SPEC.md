# Timbre — Audio Similarity Search on a Hand-Written HNSW Index

> **How to use this file:** Drop it in the repo root as `SPEC.md`. It is the source of truth for scope.
> Work one phase at a time and do not start a phase until the previous phase's **Done when** is met.
> The Guardrails section is binding — if a change would violate one, stop and ask.

---

## 1. What this is

An audio similarity search engine over the Free Music Archive corpus. Give it a 10-second clip and
it returns the tracks that sound most like it, ranked. Because CLAP embeds audio and text into a
shared space, the same index also answers typed queries like `"sparse melancholy piano"` with no
extra machinery.

**The index is the project.** CLAP and FMA are how we get 2.2M interesting vectors for free. The
HNSW implementation is hand-written; `hnswlib` and FAISS appear only as benchmark baselines and
never in the serving path.

| | |
|---|---|
| Corpus | 106,574 tracks (FMA large) |
| Index size | ~2.2M vectors, 512-d float32 (~4.6 GB raw) |
| API cost | $0 — all local / free tier |
| Core build | 3–4 weekends (Phases 0–2 comfortably, Phase 3 optimistically) |

---

## 2. Architecture

```
INGEST (offline, once)
  MP3 files ──► decode ──► segment ──► CLAP ──► vector store
  93 GiB       48 kHz     10 s win    512-d    mmap .npy
               mono       1 s hop              + SQLite metadata
                                                     │
                                                     ▼
QUERY                                        ┌───────────────┐
  clip or text ─────────────────────────────►│  HNSW index   │──► ranked tracks
                                             │   (ours)      │    segment hits
                                             └───────────────┘    → track score
```

Everything left of the index is plumbing built once. Everything at and right of it is the project.

### Segmentation

Each 30-second clip is cut into 10-second windows at a 1-second hop → **21 windows per track**.

```
106,574 tracks × 21 = 2,238,054 vectors × 512 dims × 4 bytes = 4.6 GB raw
```

One embedding per track would give only 106K vectors — too small to be interesting and too coarse
to match a clip against a specific passage. Overlapping windows fix both, at the cost of a ranking
problem: a track now has 21 chances to match, so an aggregation rule is required (see Open
Questions).

---

## 3. Build plan

### Phase 0 — Pipeline (½ weekend)

Start on `fma_medium` (25,000 tracks, 22 GiB), **not** the full corpus.

- Decode audio, resample to 48 kHz mono
- Segment into 10 s windows at 1 s hop
- Embed with `laion/larger_clap_general`, pinned to a validated revision (the
  original music checkpoint was replaced after the retrieval diagnosis).
- Write vectors to a memory-mapped `.npy`, metadata to SQLite (track id, title, artist, genre, segment offset)

Make it **resumable from the first line of code**. Embedding 25K tracks takes hours and the job
will die partway through. A pipeline that restarts from zero costs an entire weekend.

**Done when:** the job can be killed at any point, restarted, and produce a byte-identical vector store.

---

### Phase 1 — Exact search + evaluation harness (½ weekend)

Brute-force cosine similarity via batched matmul. Slow, correct, and the ground truth for
everything that follows.

Build this **before** the index. The common failure is writing the ANN index first and then having
no way to tell whether it returns good results or garbage quickly.

- Hold out a fixed 1,000-query sample from the index build
- Cache exact top-100 neighbors for each
- Implement `recall_at_k(candidate_results, ground_truth, k)`

**Done when:** exact top-100 neighbors are cached for the query sample, and any candidate index can
be scored against them with one function call.

---

### Phase 2 — HNSW, written from scratch (1–1½ weekends)

The real work. Pure Python with NumPy-vectorized distance computation, on the 125K-vector medium set.
Correctness first; slow is fine at this size.

Required components:

- Layer assignment by exponential decay
- Greedy descent through upper layers
- Beam search at layer 0 with configurable `efSearch`
- Insertion with `M` bidirectional links and pruning on overflow
- **The neighbor-selection heuristic** — not naive top-`M`; see Risks
- Graph serialization to disk and reload without rebuild

**Done when:** recall@10 ≥ 0.95 against Phase 1 ground truth, and a sweep over
`M` × `efConstruction` × `efSearch` produces a recall/latency frontier.

---

### Phase 3 — Scale it until it hurts (1–1½ weekends)

Move to `fma_large`: 106,574 tracks, 2.2M vectors, 4.6 GB of raw float32.

- Python insertion will not survive this. Port the distance kernel and search loop to C++ behind
  pybind11, and measure the speedup.
- Then deliberately break it: run under a hard memory cap (`docker run --memory=2g`) so the graph
  cannot fit in RAM, and observe what random graph traversal does to a memory-mapped file.
- Try reordering nodes by graph locality (e.g. BFS order) and measure how much is recovered.

**Done when:** you can state, with numbers, how build time, query latency, and recall each change
from 125K → 2.2M vectors, and what happens at the memory cliff.

---

### Phase 4 — Demo (optional, ½ weekend)

FastAPI in front of the index. One page: upload a clip or type a phrase, get five results with an
audio player. Deliberately last — least technically interesting, most likely to eat time.

**Done when:** a stranger can paste in a phrase, hear five results, and understand what happened.

---

## 4. Open questions

These are the point of the project. None has a published answer for this corpus; each is settled by
an experiment. Record results in the README.

1. **Do published ANN tradeoffs hold for music embeddings?** Nearly every HNSW benchmark uses SIFT,
   GIST, or text embeddings. CLAP music vectors have their own intrinsic dimensionality and cluster
   structure — genres are not uniformly dense. Whether the standard recall-vs-`efSearch` curve looks
   the same here is open.
2. **How should 21 segment hits become one track score?** Max-pool favors tracks with one strong
   moment; mean favors consistency; counting hits in the top-k favors broad matches. These rank
   music differently and no literature settles it.
3. **Does insertion order damage the graph?** FMA is ordered by track ID, which correlates with
   album and genre, so a naive build inserts thousands of similar vectors consecutively. Shuffle,
   rebuild, compare recall at equal parameters.
4. **What happens past the memory cliff?** HNSW search is a pointer chase. Once the graph outgrows
   RAM, every hop is a potential page fault. How steeply latency degrades, and how much
   locality-aware reordering recovers, depends on layout and access pattern.
5. **Small model + good index, or big model + bad index?** At a fixed latency budget, is retrieval
   quality better served by a stronger embedding model or a better-tuned index?

---

## 5. Measurement plan

Every claim in the README comes from this table. Ground truth is exact top-100 over the fixed
1,000-query held-out sample.

| Metric | How | Why it matters |
|---|---|---|
| recall@1 / @10 / @100 | vs. cached exact neighbors | Only measure of whether the index is correct |
| Query latency | p50 / p95 / p99, single-threaded | Means hide tail behavior, and the tail is the story |
| Throughput | queries/sec, 1 and N threads | Shows whether search parallelizes or contends |
| Build time | wall clock, by corpus size | Reveals the scaling exponent of insertion |
| Peak RSS | resident set at query time | The constraint that eventually defines the design |
| Baseline gap | vs. hnswlib and FAISS-HNSW | Honest calibration |

**On the baseline gap:** we will lose to hnswlib — it's years of SIMD tuning. The deliverable is not
beating it, it's a README section explaining precisely where the time goes and which of their
optimizations account for the difference.

---

## 6. Stack and data

| Component | Choice | Notes |
|---|---|---|
| Corpus | [FMA](https://github.com/mdeff/fma) | medium 25K / 22 GiB → large 106,574 / 93 GiB; CC-BY metadata |
| Embeddings | [`laion/larger_clap_general`](https://huggingface.co/laion/larger_clap_general) | Apache 2.0, joint audio+text space, 48 kHz input |
| Embedding compute | Colab or Kaggle free tier | Kaggle gives 30 GPU-hours/week; checkpoint every batch |
| Index | Ours — Python, then C++ | pybind11 for the hot path in Phase 3 |
| Vector store | memory-mapped `.npy` | mmap is what makes the memory-cliff experiment possible |
| Metadata | SQLite | Track titles, artists, genres, segment offsets |
| Serving | FastAPI | Phase 4 only |
| Baselines | hnswlib, FAISS | Reference only — never in the serving path |

**Disk is the real constraint.** 93 GiB of MP3s plus decoded audio will not sit comfortably on a
laptop. Stream, embed, and delete: keep vectors and metadata, discard audio for everything except
the few hundred tracks needed for the demo.

Confirm the CLAP embedding dimension with a single forward pass before sizing anything — the spec
assumes 512.

---

## 7. Guardrails

Every item below is a plausible next step, and every one will sink the timeline.

- **Don't train an audio model.** CLAP off the shelf. Fine-tuning is a different project.
- **Don't build a music app.** No playlists, no accounts, no library management.
- **Don't add product quantization.** Tempting and interesting. Phase 5 at the earliest.
- **Don't distribute it.** Single node. Sharding is a different curiosity.
- **Don't support deletion.** Immutable index. Deletion in HNSW is genuinely hard and buys nothing here.
- **Don't polish the UI.** One page, one input, five results.
- **Don't put a third-party ANN library in the serving path.** Baselines only.

---

## 8. Risks

- **Embedding throughput is the hidden cost.** 2.2M segments through CLAP is many GPU-hours. Batch
  aggressively, use fp16, and start the medium-set run before writing the index so it embeds in the
  background.
- **The neighbor-selection heuristic is where HNSW implementations go wrong.** Naively keeping the
  `M` closest neighbors produces a graph that looks fine and searches badly. Budget debugging time
  and use Phase 1 ground truth to catch it early.
- **Timeline.** 3–4 weekends covers Phases 0–2 comfortably and Phase 3 optimistically. Stopping
  after Phase 2 leaves a complete, honest project. Stopping mid-Phase-3 does not.
- **C++/Python boundary overhead can eat the win.** Cross the boundary once per query, not once per
  distance computation.

---

## 9. Target resume outcome

> **Timbre — Audio Similarity Search (Python, C++)**
> - Built an audio search engine over 106K tracks, segmenting clips into overlapping windows and
>   indexing 2.2M CLAP embeddings with a hand-written HNSW implementation; ported the search kernel
>   to C++ via pybind11 for a `N×` query speedup over the NumPy prototype.
> - Reached `0.9X` recall@10 against exact search at `X.X ms` p95 latency, mapping the
>   recall/latency/memory frontier across `N` parameter configurations and characterizing index
>   degradation under constrained memory.

Placeholders get filled with real measured numbers. Two bullets, both load-bearing, both defensible
under questioning.

---

*Corpus and licensing details verified September 2026 against the FMA repository and the LAION CLAP
model card. Vector counts and storage figures are computed from a 10-second window at a 1-second hop
over 30-second clips; adjust the hop to trade index size against temporal resolution.*
