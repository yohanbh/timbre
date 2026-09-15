# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Completed: FMA large vectorization

FMA large ingestion and final verification finished successfully on 2026-09-10
at 22:23 EDT. All 106,574 tracks were processed: 105,884 embedded, 162 failed,
and 528 too short, with zero pending. The store contains 2,144,867 populated
512-dimensional vectors. Shape, dtype, zero padding and vector norms passed;
store/database hashes are saved in `store/large/completion.json`. SQLite quick
check also passed after completion. The service exited successfully; do not
restart ingestion unnecessarily. Large-store ground truth is also complete (see below); index benchmarks
remain subsequent work. Windows disk compaction completed successfully.

## Completed: Windows disk space reclamation

Offline compaction completed on 2026-09-10 at 22:39 EDT, reclaiming
118,753,329,152 bytes (110.60 GiB). Windows reported 206.42 GiB free afterward.
The result and DiskPart success output were verified in
`C:\Users\bhojw\timbre-maintenance`; the current VHD size confirms the reduction.
After reopening WSL, SQLite quick check passed and ingestion counts still match
`store/large/completion.json`, with zero pending. The obsolete
`timbre-reclaim-reminder.service` is disabled and stopped. Ingestion was not restarted.

## Completed: FMA large ground truth

On 2026-09-11, `store/large/groundtruth.npz` was built and validated for 1,000
fixed queries and top-100 neighbors over 2,144,867 populated vectors. Both
self-excluded and query-track-excluded variants reproduce byte-for-byte.
`store/large/index_groundtruth.npz` freezes 2,143,867 index candidates, excluding
all 1,000 query rows, with a separate exact top-100 oracle. Source hashes,
query sampling, membership, exclusions and score ordering passed validation;
three independent per-query full scans also checked the index oracle.
Reports and hashes are in `docs/large_groundtruth_results.json` and the
`store/large/*groundtruth_validation.json` files. No large graph has been built.
Next: use the held-out oracle to build and benchmark the large C++ HNSW index;
do not re-vectorize or use the full-store oracle to score a held-out graph.
