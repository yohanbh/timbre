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
| Cache **both** unfiltered and sibling-excluded ground truth | Measured: 73.5% of top-10 are same-track siblings. One list cannot serve both index recall and retrieval quality. |

## Phase 1 — exact search + evaluation harness — DONE

Built and committed. `store/groundtruth.npz` caches exact top-100 for a fixed
1,000-query genre-stratified sample; any candidate index scores against it with
one call to `recall_at_k`.

```bash
PYTHONPATH=src python3 -m timbre.build_groundtruth   # ~20 s, rebuilds the cache
PYTHONPATH=src python3 -m pytest tests/test_groundtruth.py -q
```

**Two ground-truth variants are cached, not one.** Measured on this corpus,
73.5% of unfiltered top-10 neighbours are windows of the query's own track and
40.2% of queries have an all-sibling top-10 (within-track similarity 0.979).

| Array | Use |
|---|---|
| `gt_ids` / `gt_sims` | True nearest neighbours. The honest target for **Phase 2 index recall** — an exact index must reproduce exactly this. |
| `gt_ids_nosib` / `gt_sims_nosib` | Same-track windows removed. The target for **retrieval quality** and the aggregation experiment, where "found the same song again" is not the question. |

The two overlap only 0.265 at k=10, so the choice is not cosmetic: scoring the
aggregation rule against the unfiltered list would mostly measure self-recall.

### What Phase 1 established

- Query sample is reproducible from `SEED` alone; all 16 genres present, one
  window per track (siblings are near-duplicate queries and buy less than a
  different track for the same cost).
- Reads go through `load_layout`, never a bare arange: the store is allocated at
  21 windows/track but most fill 20, so the raw array has zero-filled holes.
- Vectors are L2-normalized at ingest, so cosine is a plain dot product.
- **Chunk width changes the last bits.** The same dot products at chunk=1500 vs
  20000 differ on ~1.3% of elements by up to 7.7e-07, in the raw BLAS output —
  the same tile-decomposition effect `embed.py` documents for batch composition.
  Neighbour *ids* are bit-stable and are what `recall_at_k` consumes; *sims* are
  only float32-stable. Do not assert bit-identical sims across chunk widths.

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
