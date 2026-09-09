# Timbre

Audio similarity search over the Free Music Archive, on a hand-written HNSW index.
See `TIMBRE_SPEC.md` for scope. **The index is the project**; Phase 0 is the pipeline
that feeds it.

## Phase 0 — Ingest pipeline

```
mp3 ──► ffmpeg decode ──► 21 windows ──► CLAP ──► mmap .npy
        48 kHz mono       10 s @ 1 s hop  512-d    + SQLite
```

```bash
uv sync
python -m timbre.ingest --audio-root data/fma_medium   # resumable; just rerun after a crash
python -m timbre.verify                                 # hashes + integrity checks
bash tests/kill_restart.sh <audio_root> <db> <store> 5  # the acceptance gate
```

### Measured on this machine (RTX 3060 6 GB, 16 cores, 7.4 GiB RAM)

| Stage | Cost per track |
|---|---|
| ffmpeg decode | 113 ms |
| Mel extraction (1 core) | 1000 ms |
| CLAP forward, batch of 21 | 159 ms (132 windows/s) |

With 12 workers the CPU pool supplies 10.8 tracks/s against 6.3 tracks/s of GPU
capacity, so the pipeline is GPU-bound by design — the irreducible cost. Without
the pool a single CPU feeds only 0.9 tracks/s and the GPU idles ~85%. Peak VRAM
1475 MiB, peak RSS 2.1 GiB. Projected full-corpus run: **~1.1 h for 25,000 tracks**.

These figures come from synthetic constant-bitrate audio. Real FMA mp3s are VBR
and decode more slowly, so treat 1.1 h as a floor rather than a prediction.

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

Consequence: ~505K real vectors rather than 525K, and "21 windows per track" in the
spec is really "20 or 21".

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
