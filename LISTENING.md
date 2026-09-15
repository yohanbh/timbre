# Listening evaluation

Human judgments of musical similarity, kept deliberately separate from geometric
index recall. A 99.58% recall@10 means the index faithfully reproduces exact
cosine search; it says nothing about whether the retrieved audio sounds alike.
This file records what listening established.

All results below are on the **medium store** (24,980 tracks, 510,064 vectors,
`laion/larger_clap_general`). Session date 2026-09-14.

## Tools

```bash
# Automated: artist self-retrieval lift over chance.
USE_TF=0 PYTHONPATH=src python3 tests/artist_retrieval.py

# Interactive: blind A/B, rank 1 vs rank 40, with replay.
USE_TF=0 PYTHONPATH=src python3 tests/blind_ab.py --trials 20
```

`tests/blind_ab.py` plays the query passage, then two candidates in random
order, and never reveals which is which until the summary. Answer `1`/`2`,
`r` to replay the whole trial, `s` to skip, `q` to stop. It reports a one-sided
binomial p-value and splits the score by whether a trial was replayed.

## 1. Temporal resolution is real

**Finding: querying different parts of one track returns genuinely different and
individually appropriate results.**

Measured over 40 sampled tracks with >= 20 windows, comparing top-10 at offset 0
against the last window:

| Metric | Value |
|---|---|
| Mean top-10 overlap | 2.8/10 |
| Median overlap | 2/10 |
| Identical result sets | 0/40 |
| Fully disjoint sets | 11/40 |

Within-track geometry on the replacement checkpoint, over 300 sampled tracks:

| Metric | Replacement | Original (README historical) |
|---|---|---|
| Mean within-track pairwise cosine | **0.917** | 0.979 |
| First window versus last window | **0.821** | not measured |

Listening confirmation, track 4309 "Useless" by The Cynics (Rock, 20 windows,
0/10 overlap): the early passage is horn instrumentation, the late passage is
guitar with low-key vocals. Both result sets matched their own passage. The late
query returned another Cynics track, "I Want You", at rank 3, which the listener
confirmed sounds like the late section specifically.

**Consequence.** The 21-windows-per-track storage earns its keep, and the
open max-pool versus mean versus count-in-top-k aggregation question is a real
experiment. The README's "a track's 20 windows are nearly one vector" was an
original-checkpoint measurement and no longer describes the store.

## 2. The production-era hypothesis is retired

**Finding: no evidence the model ranks by recording character over content.**

The README listed "keying on encoding/production" as inconclusive. The specific
prediction was that lo-fi recording character would outrank musical content.

Control: track 23190 "Titanic Woompah" by Sister Gwen Mckay (Spoken), top score
0.601 against 0.92+ elsewhere, so the model signals low confidence on an isolated
query.

- Rank 1 is the query artist's only other track in the store. The listener
  confirmed it is a **different speaker**, so this is a same-artist hit rather
  than a same-voice hit.
- Ranks 2-8 are a mix of speech and early-1900s acoustic recordings. The listener
  reported "a good mix of both" rather than noise dominating.
- The decisive negative: **no case of singing in a crackly old recording ranked
  above clean modern speech.** That was the production-over-content signature,
  and it did not appear.

## 3. Artist self-retrieval measures musical similarity, not just genre

**Finding: median 39x lift over chance, and the lift reflects real musical
similarity rather than genre-plus-production artifacts.**

Artists with >= 20 tracks, 5 random query tracks each, hits in top-20 against the
share expected by chance. Full results in `docs/listening_artist_retrieval.json`.

| Artist | Tracks | Hits@20 | Expected | Lift |
|---|---|---|---|---|
| Derek Clegg | 75 | 17.8 | 0.06 | 296x |
| Squire Tuck | 121 | 12.8 | 0.10 | 132x |
| Big Blood | 112 | 8.6 | 0.09 | 96x |
| Blue Dot Sessions | 213 | 7.8 | 0.17 | 46x |
| Ergo Phizmiz | 112 | 2.2 | 0.09 | 25x |
| Cullah | 81 | 1.6 | 0.06 | 25x |

Median lift across 14 artists: **39.2x**.

Two cases were checked by ear:

**Derek Clegg, track 126746 "Best Of Me".** Ranks 1-7 are all Albin Andersson,
zero Clegg. The listener judged the voices and instrumentation "VERY SIMILAR"
with very similar styles, and the result a good match. So the model found the
nearest available music and was correct to; this is not a failure to retrieve
Clegg's 74 siblings.

**Ergo Phizmiz, track 24977 "Peanuts" (Experimental).** Eight results spread over
five genres with scores bunched from 0.790 to 0.765. Tightly bunched scores can
mean nothing is genuinely close, but the listener heard a real common thread --
horns, low-key mellow singing, funk, samba/reggae feel -- and rated the
neighbourhood "reasonable, but it could be better."

## 4. Blind A/B: inconclusive overall, suggestive by confidence

**Finding: the headline result is null and underpowered. The interesting signal
is that listener accuracy tracks the model's own confidence.**

19 scored trials, rank 1 versus rank 40, seed 1913937587. Raw log in
`docs/listening_blind_ab_20260914.json`.

| Result | Value |
|---|---|
| Score | 11/19 (58%) |
| One-sided binomial p | 0.3238, not significant |
| Replayed trials | 9/19, scoring 5/9 |
| First-listen trials | 10, scoring 6/10 |

**The design was underpowered.** 19 trials need 14/19 for p < 0.05. Against a
true 65% skill level this design had roughly **30% power** -- more likely to miss
a real effect than to find one. "Not significant" here means "could not tell",
not "no effect".

| Trials | Power vs true 65% |
|---|---|
| 19 | 30% |
| 40 | 57% |
| 80 | 85% |
| 100 | 91% |

**The post-hoc split.** Dividing trials at the median cosine gap between the
rank-1 and rank-40 candidates:

| Subgroup | Accuracy |
|---|---|
| Gap >= 0.064 (clearer pairs) | 8/10 (80%) |
| Gap < 0.064 (closer pairs) | 3/9 (33%) |

This matches the listener's own report of hearing a clear difference on some
trials and coin-flipping on others. **Treat it as a hypothesis, not a finding:**
the cutoff was chosen after seeing the data, and the subgroup is p = 0.055 at
n = 10. It needs a prospective test.

Replaying did not help (5/9 versus 6/10 first-listen), consistent with those
trials being genuinely ambiguous rather than mis-heard.

## Open

- **Prospective gap-stratified blind A/B, ~40 trials.** Sample candidate pairs
  across a designed range of cosine gaps instead of always rank 1 versus rank 40.
  This tests the 80%/33% split at adequate power and locates the similarity
  threshold below which the index's ranking stops being audible -- more useful
  than a single accuracy number.
- Aggregation experiment: max versus mean versus count-in-top-k, now that
  temporal resolution is established.
- Repeat the artist and temporal probes on the large store (105,884 tracks,
  248 artists with 50+ tracks) to see whether findings survive 4.2x scale.
