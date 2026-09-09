# Next steps

Phase 0 is complete and committed. This file is the pickup point.

## Where things stand

**Phase 0 — DONE.** `store/vectors.npy` (1,075,200,128 bytes) holds 510,064 real
CLAP vectors for 24,980 of 25,000 fma_medium tracks (15 corrupt, 5 too short).
`store/timbre.db` has full metadata. The phase gate is met: killed mid-run at full
scale, restarted, byte-identical store (`664ef3c0...`) and bookkeeping
(`8f52a7a4...`).

Rebuild from scratch with:

```bash
PYTHONPATH=src python3 -m timbre          # ~1.4 h, resumable, run again after any crash
PYTHONPATH=src python3 -m timbre.verify   # hole integrity + norms + hashes
```

Note it is `python -m timbre`, **not** `python -m timbre.ingest` -- the entrypoint
pins BLAS/OpenMP thread limits before numpy and torch are imported. Bypassing it
costs ~6x throughput (see README).

## Decisions already made (do not relitigate)

| Decision | Why |
|---|---|
| Index the **raw** space, no mean-centering | Tested: +1.1% at n=2000, noise. Centering preserves ranking, and ranking is all retrieval uses. |
| Strict 480,000-sample windows; accept 20 windows | 59% of clips are 29.9766 s (encoder artifact). Padding would add a near-duplicate of window 20 plus silence. |
| No `segments` table | Derivable (`offset_s = row_id - row_start`), and a second table consistent with the mmap would break the atomic commit. |
| 6 workers | GPU saturates at 91-97%; more only adds memory pressure on a 7.4 GiB box. |

## Phase 1 — exact search + evaluation harness (½ weekend)

Build this **before** the HNSW index. Without ground truth there is no way to tell
a working index from a broken one, and the spec's own risk note says a naive
neighbour-selection heuristic produces a graph that looks fine and searches badly.

```
1. Chunked brute-force cosine over 510,064 vectors
   -> verify: matches a naive loop on 100 sampled queries
2. Fixed 1,000-query held-out sample, genre-stratified
   -> verify: excluded from the index build; selection is reproducible from a seed
3. Cache exact top-100 neighbours per query
   -> verify: reload gives identical neighbour ids
4. recall_at_k(candidate_results, ground_truth, k)
   -> verify: returns 1.0 when scored against ground truth itself
```

**Done when:** exact top-100 is cached for the query sample, and any candidate
index can be scored against it with one function call.

### Three choices to make, with the evidence behind them

- **Segment-level ground truth, not track-level.** Within-track windows are 0.979
  similar, so track-level mean-pooling discards most of what the 21x storage paid
  for. Building ground truth on pooled vectors would bake that loss into every
  Phase 2 measurement. Track scores derive from segment scores; not the reverse.
- **Genre-stratified query sampling.** The corpus is 28% Rock / 25% Electronic, so
  uniform sampling yields a query set that is mostly those two. Open question #1
  is partly about whether density varies by genre, which needs coverage.
- **Chunk the matmul from the start.** 510K x 512 is ~1 GB per full pass; fine for
  1,000 queries, but chunking is barely more code and avoids a rewrite later.

## Phase 2 — HNSW from scratch (1-1½ weekends)

The actual project. Per spec: layer assignment by exponential decay, greedy
descent, beam search at layer 0 with configurable `efSearch`, insertion with `M`
bidirectional links and pruning, **the neighbour-selection heuristic** (not naive
top-`M`), and graph serialization.

**Done when:** recall@10 >= 0.95 against Phase 1 ground truth, plus a sweep over
`M` x `efConstruction` x `efSearch` producing a recall/latency frontier.

Budget real debugging time for the neighbour-selection heuristic. The weak
geometric signal in this corpus (see README) means a bad graph will look
plausible, so lean on ground truth early rather than eyeballing results.

## Open experiments worth running

These are cheap and each settles a spec open question:

1. **Aggregation rule (open question #2).** Compare max-pool vs mean vs
   count-in-top-k for turning 20 segment hits into one track score. The 0.979
   within-track similarity suggests mean is too blunt; this is now an empirical
   question with a clear setup.
2. **Insertion order (open question #3).** FMA is ordered by track ID, which
   correlates with album and genre, so a naive build inserts thousands of similar
   vectors consecutively. Shuffle, rebuild, compare recall at equal parameters.
3. **Text-query quality.** `tests/listen.py --text "..."` works but retrieval is
   weak (~0.07 similarity, poorly separated). If Phase 4's demo matters, this
   needs calibration -- possibly per-query score normalization.

## Tools already built

```bash
# retrieval quality: raw vs centered, genre@10 against 17.2% chance
PYTHONPATH=src python3 tests/centering_test.py 2000

# listen to a query and its neighbours (needs ffplay)
PYTHONPATH=src python3 tests/listen.py --track 2 -k 5 --play
PYTHONPATH=src python3 tests/listen.py --text "smooth jazz saxophone" -k 5

# determinism + crash-window regression tests
python3 -m pytest tests/test_determinism.py -q          # fast subset
python3 -m pytest tests/test_determinism.py -q -m slow  # includes GPU

# full-scale kill-restart gate
bash tests/kill_restart.sh <audio_root> <db> <store> 5
```

## Known gaps

- Byte-identity is promised only within a fixed environment. Library versions and
  the model commit SHA are recorded in the `meta` table and a resume against a
  changed environment is refused -- but a torch or driver upgrade means the store
  must be rebuilt to stay bit-comparable.
