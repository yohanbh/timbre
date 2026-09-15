# FMA large ingestion

FMA large ingestion and final verification completed successfully on 2026-09-10
at 22:23 EDT. All 106,574 tracks were processed: **105,884 embedded**, 162 failed,
528 too short and **zero pending**. The store contains **2,144,867 populated
512-dimensional vectors**. Shape, dtype, zero padding and vector norms passed;
SQLite quick check passed after completion. Model identity and store/database
SHA-256 hashes are recorded in `store/large/completion.json`. The supervised
service exited successfully and no ingestion workers remain running.

Windows disk compaction completed at 22:39 EDT, reclaiming 110.60 GiB; Windows
reported 206.42 GiB free afterward. DiskPart success, VHD shrinkage and the
post-compaction SQLite checkpoint were verified. The reminder service is disabled.
Large-store ground truth and the held-out index oracle were completed and
validated on 2026-09-11; see [PHASE3.md](PHASE3.md#large-store-ground-truth--2026-09-11).
Large-index construction and benchmarks remain next.

Historical extraction and restart details follow.
`store/large/extraction_limits.json` records the change. Extraction completed at
15:21 EDT and all 106,574 track IDs were verified. The supervised handoff succeeded;
All 24,980 completed medium tracks were reused with zero audio mismatches, leaving
81,594 tracks for the original ingestion run. Use the status files below
for live progress and final completion.

The user authorized deleting both audio ZIPs after the restart path was updated
and checked. `data/fma_large.zip` and `data/fma_medium.zip` have been removed
(115.61 GiB combined). Extracted audio and metadata remain. Restarting now uses
the verified extraction marker and checks complete track-ID coverage, so the
archives are no longer required. Details are in `store/large/archive_cleanup.json`.

The [official FMA release](https://github.com/mdeff/fma) contains 106,574 tracks
in the large subset, with 30-second excerpts. The archive is
[fma_large.zip](https://os.unil.cloud.switch.ch/fma/fma_large.zip),
100,306,112,191 bytes (93.42 GiB), with published SHA-1
`497109f4dd721066b5ce5e5f250ec604dc78939e`.

## Automatic stages

`scripts/ingest_fma_large.sh` supervises the following sequence:

If `store/large/extraction.paused` exists, the script exits before any archive work.

When `store/large/extracted.sha1` exists, startup checks its published checksum
value and validates the extracted track IDs against metadata, then skips download
and archive access. An invalid marker or missing tracks stops the run before
vectorization. Without a marker, the original download/verify/extract path below
creates a fresh verified corpus.

1. Resume the download into `data/fma_large.zip.part`, retrying network failures.
2. Verify the published archive checksum, then rename it to `data/fma_large.zip`.
3. Extract to `data/fma_large` and verify all 106,574 track IDs against metadata
   before allocating the manifest. Extraction checks ZIP CRCs through `unzip`.
   If `store/large/vectorization.paused` exists, stop here with stage
   `awaiting_vectorization_approval`. The remaining stages require explicit user
   instruction to lift the hold.
4. Create `store/large/timbre.db` and `store/large/vectors.npy`. Validate the medium
   store against its saved ground-truth identity and integrity checks. Reuse its
   completed embeddings only where the embedding configuration and MP3 SHA-256
   match. Remap rows by track ID; flush and fsync vectors before committing markers.
   Up to 24,980 tracks can be reused, leaving at least 81,594 to process.
5. Embed pending tracks with the existing pinned `laion/larger_clap_general` model
   and six CPU preparation workers feeding the GPU. Each track is checkpointed.
6. Require zero pending tracks and verify array shape, float32 type, zero padding
   and vector norms. Write counts, model identity and hashes to `completion.json`.
   Corrupt or short tracks are recorded separately and retain zero-filled slots.

The allocation is 2,238,054 rows × 512 float32 values (about 4.27 GiB). The number
of populated vectors will be smaller and is only known after ingestion. Existing
medium data, vectors, ground truth and index artifacts are preserved. Large-store
ground truth is complete; large-index construction and benchmarks remain next.

## Progress

Run from `/home/yohanb/timbre`:

```bash
# Completed successfully; ingestion service should show active (exited).
systemctl --user status timbre-fma-large-download timbre-fma-large-ingest timbre-fma-large-extraction-hold --no-pager

# Current pipeline stage and log; Ctrl-C stops following, not ingestion.
cat store/large/stage
tail -f store/large/pipeline.log

# Track counts once the large manifest has been created.
python3 -c 'import sqlite3; c=sqlite3.connect("file:store/large/timbre.db?mode=ro",uri=True); print(dict(c.execute("SELECT status,COUNT(*) FROM tracks GROUP BY status")))'

# Written only after final verification succeeds.
cat store/large/completion.json
```

`store/large/stage` moves through `downloading`, `verifying_archive`, `extracting`,
`reusing_medium_vectors`, `embedding`, `verifying_vectors`, and `complete`.
Archive-free restarts begin with `verifying_extraction`.
The temporary handoff also records `starting_ingestion`. If the user requests
another hold, pause stages can include `extraction_paused`, `vectorization_paused`, and
`awaiting_vectorization_approval`. On a
normal error it records `failed:<stage> (exit <code>)`; service status and the log
are authoritative if the process is killed. Download errors are logged separately
in `data/fma_large_download.log`. Successful services show `active (exited)`.

## Interruption and resume

The user systemd services survive a chat disconnect. Keep the machine and WSL
running; they cannot perform work while the host is asleep or shut down. The
download supports HTTP resume, and ingestion skips committed tracks.

The handoff completed at 15:21 EDT. The original stopped Bash supervisor and its
extraction children have exited; their PIDs in `extraction_pause.json` are historical
and must not be resumed. `timbre-fma-large-extraction-hold` completed successfully,
restarting `timbre-fma-large-ingest` with a fresh Bash process. The latter owns all
remaining archive checks, vector reuse, GPU ingestion and final verification.

Both `MemoryHigh` and `MemoryMax` have already been restored to `infinity` for
`timbre-fma-large-ingest.service`; the earlier 768 MiB extraction limit must not
be applied to GPU ingestion. The latest user request lifted the hold with no resource constraints.
`vectorization.paused` was removed and the transient ingestion service recreated
with the relaunch command below. Completed tracks are skipped and unfinished work is retried. The
disk-space reminder remains waiting for successful final verification.

After correcting an ingestion failure, while the service still exists:

```bash
systemctl --user restart timbre-fma-large-ingest.service
```

After a WSL shutdown/restart, the transient services no longer exist. Relaunch:

```bash
mkdir -p store/large
systemd-run --user --unit=timbre-fma-large-ingest \
  --description='Verify, extract and vectorize FMA large' \
  --property=WorkingDirectory=/home/yohanb/timbre \
  --property=RemainAfterExit=yes \
  --property=StandardOutput=append:/home/yohanb/timbre/store/large/pipeline.log \
  --property=StandardError=append:/home/yohanb/timbre/store/large/pipeline.log \
  /usr/bin/bash /home/yohanb/timbre/scripts/ingest_fma_large.sh
```

The script validates an already verified extracted corpus without needing ZIPs.
It uses the download path only when there is no extraction marker. A file lock
prevents two ingestion pipelines from writing the same store.

## Preflight evidence

Before queuing ingestion, all three GPU determinism tests passed, including the
crash-window rewrite test. Fresh GPU embeddings of tracks 2, 3 and 10 matched the
completed medium store byte for byte. Reuse tests cover row remapping, zero padding,
changed audio, repeated invocation and model revision drift. The complete non-GPU
suite passed: 70 tests, with the three GPU tests run separately. Shell syntax and
Git whitespace checks also passed.

At GPU startup, the large manifest's 106,574 tracks, array shape, dtype and model
identity were verified. Three sampled reused blocks matched the medium vectors
byte for byte. Three sampled newly embedded tracks passed norm and zero-padding
checks. The service had no OOM events, both memory limits were unlimited, and
committed track counts were increasing. The complete-store checks run at the end.

Before archive cleanup, isolated restart checks used the real extracted corpus
and metadata with no ZIPs present, stopping at the vectorization hold. The valid
corpus passed; invalid-marker and missing-track cases failed before archive access
or ingestion. The script was replaced atomically so the running Bash supervisor
could finish reading its original script without interruption.
