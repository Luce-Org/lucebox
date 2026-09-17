# Cache refinements and independent audit — September 15

## Status

Updated and restarted on linuxmacan after user authorization. Production readiness and a short exact inference passed. The running executable hash matches the staged binary: `2ac45f2e391af1f3df615f644de3c2a52e6d16a5da37b4a2a3c60feda5f60a26`. Pi extension installed; existing Pi sessions need `/reload` to load its logging change. Exact remains the default, one active stream, timing projections disabled.

Rollback: `/opt/lucebox/work/hybrid-cache-20260915/deployment-before-28968c93b22d4efd8c8eecd4d3ce5e71`. The HIP library hash is unchanged from the preserved backup. Existing production calibration observations were retained.

Deployment checks: all 567 server tests passed. Focused private GPU test restored 4,608 prefix tokens during rebase (22.68 seconds total), then rejected the stale checkpoint after an earlier history edit. That edited-history request reused the frozen region with zero rescoring (12.98 seconds). These are single functional checks, not a speedup benchmark. Raw results: `work/hybrid-cache/rebase-results.jsonl`.

## Implemented in this pass

- Deliberate rebases enumerate compatible same-owner snapshots before the earliest changed message. The source stays reserved during scoring; final rendered tokens and the backend's actual saved position must match before restoration. A mismatch drops restoration and its calibration key.
- Routing supports multiple compression/rebase candidates. Measured rebuilds compete under the existing 20% and one-second savings rule.
- Optional cold-context projections use three-observation anchors, compatible settings, one scoring region, and at most 50% input growth. Dense prefill is discounted and compression cost inflated quadratically for the cold comparison. These are heuristic decision bounds, not guaranteed latency bounds. They cannot displace reusable checkpoints. Memory feasibility is calculated separately.
- `hybrid_projections` is a validated boolean, false by default. The gateway explicitly resets its backend environment setting on each load. Four measured anchors were extracted into `work/hybrid-cache/calibration-with-anchors.json`; existing latency totals were preserved. No extrapolated timing is represented as a measurement.
- Deployment now checks for active work, preserves per-attempt rollback files, and requires readiness plus an exact inference check. Failed installation restores the previous files and verifies recovery. Mock rollback tests passed; production deployment subsequently passed readiness and exact inference.
- Pi prefix-break diagnostics omit prompt excerpts by default; `PCG_LOG_TEXT=1` explicitly enables them.

## Independent review

The reviewer identified a timing-bound bug: a discounted dense estimate could displace real cache reuse. The corrected selector marks projected candidates and excludes them from reuse challenges while considering every measured rebase candidate. The reviewer found no remaining blocker in that narrow recheck. This is not a certification of all original rollout gates.

## Local validation

Passed:

- C++ routing regression executable: projected dense/compression cannot displace reuse or hide measured rebase; exact/unknown-cost handling, invalid candidates, projection limits, prefix mismatch, settings isolation, median/EWMA updates.
- Complete HTTP translation-unit syntax check with local headers.
- Five gateway tests, using Python 3.12 (system Python 3.9 lacks the existing `asyncio.timeout` API).
- Two deployment test methods covering success, failure rollback, and busy refusal.
- Pi hook tests and real SessionManager branch tests, including 50 revisits.
- Local synthetic Pi wire test: five requests, medium thinking, 32,768 output allowance, two compaction requests.
- Default-private and explicit-text logging regression.

The subsequent GPU checks above establish the exercised rebase behavior only; they do not establish general performance or semantic equivalence.

## Audit coverage and gaps

Previously explored: GPU versus RAM snapshots, rolling session checkpoints, exact/full and frozen-prefix reuse, PFlash scoring/compression, scoring window sizes, prefill batch variants, and several HIP attention/matrix paths. Existing measurements and quality losses remain in the main results report.

Still incomplete relative to the broad design:

- Memory admission uses fixed working allowances rather than a learned, phase-specific peak model.
- Optional cache utility uses a processing-time heuristic rather than measured saved time and reuse frequency.
- Completed compression results are retained through checkpoint dependencies; they do not yet have an independent cache when capture is skipped or fails.
- Cost buckets are end-to-end, not independently calibrated phase models. Projection coverage remains deliberately narrow.
- Observation mode executes a dense exact baseline, not a cache-aware exact baseline.
- Shared-block APC/PagedAttention, KVFlash, additional KV quantization, and a general quality detector are not implemented or validated here.

## Next quiet-window checks

1. Completed: backend build and all 567 tests.
2. Completed: prefix rebase and earlier-token mutation checks. Cancellation during scoring remains a follow-up check.
3. Compare a nearby intermediate size with projections off/on, checking predicted versus actual latency and memory. Keep projections off if the comparison does not validate the assumptions.
4. Deployed with projections disabled, preserving exact override and rollback. No three-chat stress run or 80K benchmark is needed under the user's revised testing scope; larger-size estimates must stay labeled extrapolations.
