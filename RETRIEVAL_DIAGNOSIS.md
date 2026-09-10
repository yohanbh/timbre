# Retrieval diagnosis — 2026-09-09

**Repair completed:** `larger_clap_general` is now active, with 510,064 rebuilt
vectors and new exact-search ground truth. The original store is preserved in
`store/legacy_music`. Full-corpus track-mean genre@10 rose from **31.5% to 70.0%**
on the same 2,000 queries. All 26 non-slow tests pass against the completed store;
the three GPU tests (text separation, determinism, crash-window recovery) passed
during the rebuild. Listening now plays the actual winning segment.

The originally pinned `laion/larger_clap_music` checkpoint was the leading cause of poor
retrieval. Its text embeddings collapse to almost one direction. The previous
conclusion that the measured geometry is simply an inherent limitation of CLAP
on FMA was premature: repeating the same embedding pipeline tests reproducibility,
not whether the checkpoint represents sound correctly.

## Reproduced locally

Using the original model revision, processor, Transformers 4.53.0 and GPU,
these four phrases yield the following cosine matrix:

1. a heavy metal guitar solo
2. a slow sad piano ballad
3. a female opera singer
4. fast electronic dance music

```text
1.000000  0.999435  0.999266  0.999009
0.999435  1.000000  0.999474  0.999219
0.999266  0.999474  1.000000  0.999365
0.999009  0.999219  0.999365  1.000000
```

The same phrases with `laion/larger_clap_general` at revision
`ada0c23a36c4e8582805bb38fec3905903f18b41` have off-diagonal cosines
between 0.0694 and 0.4401. The tokenizer and audio preprocessor files are identical
between these two cached revisions; this comparison changes the model checkpoint.

This reproduces the symptom reported in the
[model's discussion](https://huggingface.co/laion/larger_clap_music/discussions/2)
and the more explicit
[MTEB issue](https://github.com/embeddings-benchmark/mteb/issues/5069).
Those reports alone do not establish the cause of audio-to-audio weakness.
The audio comparison below tests that separately. The exact internal defect
(training, conversion, or model/config interaction) has not been isolated.

## Checks of the original pipeline

- All 15 non-slow tests pass, including chunked exact search versus a naive scan,
  self/sibling exclusion, and ground-truth cache rebuild checks.
- Full-store verification passes both hole integrity and unit-norm checks.
  The vector SHA-256 remains
  `664ef3c022f2c02d45bf5c64149a118d1c7e4b554be8f1ea4650b70efb72583f`.
- Freshly decoding and embedding one track from each of 16 genres reproduces
  the corresponding stored first windows within 4.57e-08 maximum absolute error.
  The different batch composition explains errors of this scale.
- Sampled decoded audio is nonzero and reaches the processor at 48 kHz mono.
  Embeddings are normalized inside `ClapModel.get_audio_features`.

These checks give no evidence that storage corruption or the cosine-search
implementation causes the broad mismatch. They cannot establish perceptual quality.

## Other issues affecting interpretation

The original `tests/listen.py` averaged all windows of each track and searched
those averages. Playback always started at zero, so the passage being heard
could differ from what contributed to the ranking. The repaired tool searches
one query window, groups candidate segment hits by track, and plays each winning
window at its stored offset.

Phase 1 "ground truth" means exact neighbors in the supplied embedding space.
Even perfect recall against that cache can reproduce musically poor results.
Sibling-excluded ground truth remains a geometric target, not human relevance
labels; several hits can still belong to the same candidate track.

Several earlier README interpretations should not be carried forward:

- A fresh embedding matching stored bytes does not prove checkpoint correctness.
- Centering followed by L2 normalization can change cosine rankings. The small
  measured genre improvement is specific to the old embeddings; invariance of
  Euclidean distances under translation is a different statement.
- High within-track cosine does not establish that averaging loses no useful
  information when unrelated tracks also have high cosine and adjacent windows
  overlap by 90%.
- Genre agreement exceeding bitrate agreement does not rule out production or
  encoding confounds; those labels have different distributions.
- Changing score scale or applying a monotonic normalization per query cannot
  improve its ranking or recover distinctions missing from the embeddings.

## Reproduce the audio comparison

Measured on the same 512 tracks and exact corresponding windows:

| Metric | Stored `larger_clap_music` | Fresh `larger_clap_general` |
|---|---:|---:|
| Same genre among top 10 other tracks | 25.31% | 57.40% |
| Mean cosine between different tracks | 0.8585 | 0.3245 |
| Norm of mean embedding | 0.9267 | 0.5708 |

Chance genre agreement for this candidate pool is 18.07%. The improvement is
32.09 percentage points, using unchanged audio preprocessing and cosine search.
This strongly implicates the pinned music checkpoint in the weak audio retrieval;
it is not merely a failure confined to its text tower. Results are saved in
`store/quality_general_512.npz`.

```bash
PYTHONPATH=src python3 tests/embedding_quality.py \
  --db store/legacy_music/timbre.db --store store/legacy_music/vectors.npy \
  --revision ada0c23a36c4e8582805bb38fec3905903f18b41 \
  --out store/quality_general_512_rerun.npz
```

The script picks 512 tracks uniformly without replacement, with seed 20260909,
and one random valid ten-second window per track. Both representations search
the same 512 candidates, excluding self. Existing vectors come from the exact
corresponding stored rows. New vectors use the same decoder and window samples.
Genre precision is a weak relevance proxy; it is not a listening result or an
estimate of precision over the full 25,000-track corpus.

Use `--model laion/larger_clap_music --revision
a0b4534a14f58e20944452dff00a22a06ce629d1` to repeat the original checkpoint's
text sanity check and compare its fresh audio vectors to the stored baseline.

The diagnostic reads the production database and vectors without modifying them.
Its optional NPZ records track IDs, window row IDs, embeddings, text probes,
model revisions, and aggregate metrics.

## Completed repair and validation

- Model and processor pin: `laion/larger_clap_general`, revision
  `ada0c23a36c4e8582805bb38fec3905903f18b41`. A startup text-separation check
  rejects the original collapse symptom. Text queries check store compatibility.
- All 24,980 usable tracks rebuilt. The 15 failed and five short tracks, window
  counts, and row assignments match the original corpus exactly. Both hole and
  norm checks pass; sampled rebuilt vectors match the validated 512-track pilot
  within 5.97e-08, consistent with different batch composition.
- New vector SHA-256:
  `67bfc23b7f26fe97c0751192be478ebfea50464462c334ed44cbc9af7026d3da`.
- Ground truth rebuilt for the 1,000 fixed queries, with vector and manifest
  fingerprints. `load_groundtruth` rejects stale or legacy caches and incomplete
  ingests. Cache rebuilds reproduce IDs and scores in the tests.
- On 2,000 fixed queries against all tracks, raw track-mean genre@10 is 70.0%
  versus 31.5% originally. Centering gives 70.2%, so raw vectors remain the default.
  The measurement is recorded in `store/general/quality_full.log`.
- Queries 2 ("Food") and 3 ("Electric Ave") now each return five Hip-Hop tracks
  when searching their first ten seconds. Query 10 ("Freeway") now returns
  "Teenager" by Apache Dropout at rank five, matching its 19–29s segment. The old
  fifth result was "No Regret" by Holy Pain. Full result lists are saved in
  `store/general/listening_examples.json` and `store/old_listening_examples.json`.

These examples are ready for human listening; metadata and cosine scores alone
do not establish similarity in instrumentation, rhythm, or mood. Use:

```bash
PYTHONPATH=src python3 tests/listen.py --track 2 -k 5 --play
PYTHONPATH=src python3 tests/listen.py --track 10 -k 5 --play
```

After auditioning the repaired retrievals, the user reported that the similarity
was substantially better. The quantitative proxy and the listening feedback now
support proceeding to Phase 2.
