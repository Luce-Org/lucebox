# PFlash MVP Ship Plan — Adaptive Keep_Ratio Bandit

**Branch:** `feat/pflash-mvp-adaptive-keep` (fresh from `origin/main` @ `538bf53`)
**Ship target:** 5–7 days
**Author state:** anchored, post-chronos review

## The MVP, in one sentence

The existing pflash drafter mechanism, with **per-session adaptive keep_ratio** tuned by **DFlash chain accept-rate feedback**, exposed as a **no-knob HTTP API**. No new compression mechanism. No skip+anchor. ~220 LOC, one PR.

That's it.

## Foundations (what chronos confirmed is solid)

These are committed-with-evidence and form the substrate this PR ships on top of:

| Foundation | Commit | What it gives us |
|---|---|---|
| TDD-fixed PFlashMode wiring | `8bb77e0` | `OFF/AUTO/ALWAYS` per-request override, anchor recall regression closed, 400-on-bad-mode |
| 48-cell NIAH envelope (4K-32K) | `e3cd31f` | 100% accuracy at every (ctx × keep × mode) — **keep_ratio has free latitude in [0.025, 0.20] at ≤32K** |
| DFlash chain composition | `51c8763` | 3/3 multi-turn OK_DONE under real compression — **DFlash accept_rate is the reward signal the bandit will read** |
| Empirically-validated defaults | `8cc870a` | `L_compress=32768`, `threshold=32000`, `keep_ratio=0.05` — the priors the bandit starts from |
| 64K stability + DFlash multi-turn | `8707f25` | server runs to 128K in 23.5 GB; 64K agentic multi-turn 3/3 OK_DONE |
| 168-turn anchor coverage | `6c8e88d` | per-bucket anchor-zero distribution; informs whether bandit needs anchor-aware behavior |
| Codex adaptive keep_ratio design | `879ce95` (file `thoughts/2026-05-21-pflash-adaptive-keep-ratio-design.md`) | the 9-section design doc — concrete file:line touchpoints for the 220-LOC PR |

## Known limits that this PR does NOT pretend to fix

Honesty per chronos:

- **MTP + PFlash compose crash on turn 2+** (P0 in evidence branch, Codex investigating). Bandit reward signal will come from **DFlash chain only**; MTP path stays disabled until fixed.
- **NIAH single-needle fails at 64K+** (cliff-fix sweep `2386c2a` proved no chunk_size/anchor_radius/max_hits combo restores it; root cause is anchor-matches-on-keys-not-values). This is a **synthetic-NIAH-class limit**, not an agentic-coding limit — agentic synthesis works from kept chunks. **Document explicitly; do not ship NIAH-quality claims above 32K.**
- **Hermes harness config gap** (needs ≥64K context, today configured at 16K). Validate on claude_code + opencode only this week.
- **Opencode -0.15 ALWAYS-vs-OFF delta** (tool-loop variance, unattributed). Track but don't block.

## What this PR explicitly does NOT include

| Tempting but DROP for this ship | Reason |
|---|---|
| Skip+anchor (the `pflash_mode=always` path) | Already exists on evidence branch as opt-in; not what mrciffa asked for |
| H2 multi-resolution 2+4-gram C++ port | Validated on paper; ship later |
| H1 cosine backstop | Demoted to research-only |
| Compressed-prefix KV cache | Big feature, separate PR |
| Hybrid scorer (Momus's #1) | v2 territory |
| 64K NIAH cliff fix | Synthetic-class problem; documented limit |
| MTP re-init fix | Codex's P0, not ours this week |
| Paper draft / scaling roadmap | Brainstorm, not ship |
| vLLM portability | Distribution play; not MVP |

If any of these creeps in, it's drift. Reject.

## The 220 LOC

Per Codex's design doc (`thoughts/2026-05-21-pflash-adaptive-keep-ratio-design.md`), the change splits into:

1. **`GenerateResult.accept_rate` scalar field** (~30 LOC) — `dflash/src/common/model_backend.h` + DFlash chain populator at `qwen35_backend.cpp:932`. The MTP path populator at `:1225` is skipped this week.
2. **`AdaptiveKeepRatioState` + `step_adaptive_keep_ratio()`** (~50 LOC) — new file `dflash/src/server/adaptive_keep_ratio.h`. Pure function. Token-weighted EMA, step 0.005, bounded [0.025, 0.20].
3. **`HttpServer::sessions_` map** (~80 LOC) — `std::unordered_map<std::string, AdaptiveKeepRatioState>` guarded by mutex. Keyed by `extra_body.session_id` (parsed in `route_request`).
4. **Integration hooks** (~30 LOC) — `http_server.cpp:510` (pre-compress: read state → set `creq.keep_ratio`), `:675` (post-generate: `step_adaptive_keep_ratio(state, result.accept_rate)`).
5. **One log line per turn** (~5 LOC) — `[pflash-bandit] session=<id> turn=<n> keep=<old>→<new> (accept=<a>, ema=<ema>)`
6. **One fake-backend integration test** (~30 LOC) — `dflash/test/test_adaptive_keep_ratio.cpp`. Verifies turn-2 uses an updated ratio.

## Day-by-day plan

### Day 1 — `GenerateResult.accept_rate` plumbing
- Field added to `GenerateResult` struct
- DFlash chain populator wired at `qwen35_backend.cpp:932`
- Unit test: `/v1/messages` non-streaming response carries `usage.accept_rate` as float
- **Exit gate**: curl a single request, see `accept_rate` in the JSON response

### Day 2 — State + bandit function
- `adaptive_keep_ratio.h` with pure function + state struct
- `HttpServer::sessions_` member + mutex
- `session_id` parsed from `extra_body` in `route_request`
- Unit test: synthetic 10-turn sequence drives expected EMA + step
- **Exit gate**: state machine evolves correctly on a synthetic input

### Day 3 — Integration hooks + observability
- Pre-compress lookup at `:510`, post-generate update at `:675`
- Log line per turn
- Per-session JSONL trace to `/tmp/pflash_bandit/<session_id>.jsonl`
- **Exit gate**: 3-turn curl-driven session shows keep_ratio actually shifting

### Day 4 — Harness validation: claude_code
- `run_backend_pair.sh CLIENT=claude_code` × {fixed keep=0.05, fixed keep=0.20, bandit-default starting at 0.10}
- Compare per-turn accept_rate, total session wall, OK_DONE
- **Exit gate**: bandit Pareto-dominates at least one fixed setting on ≥ 2 of 3 sessions

### Day 5 — Harness validation: opencode
- Same A/B on opencode (tool-loop). Hermes skipped (config gap).
- Cross-client compare: does the bandit converge to similar regions?
- **Exit gate**: no client crashes; observable per-session keep_ratio trajectory committed

### Day 6 — PR prep
- `pflash/README.md` update with no-knob behavior + `session_id` opt-in
- `--help` text: `--prefill-keep-ratio` becomes the bandit's *initial prior* (additive, not breaking)
- PR description with A/B data, bandit formula, test plan
- **Exit gate**: PR opened against `main` with green CI

### Day 7 — Buffer + ship
- One regression chase
- Review comments
- **Exit gate**: mergeable

## Bail conditions

| Risk | Detection | Bail |
|---|---|---|
| DFlash accept_rate extraction is messier than expected (stderr scraping required) | Day 1 stderr inspection | Use a smaller log-grep PR first to extract reliable signal; defer bandit by 1 day |
| Bandit oscillates between bounds on real harness | Day 4 traces | Tighten step from 0.005 to 0.0025 OR widen EMA window per Codex's design |
| Cross-client variance too high | Day 5 cross-client compare | Per-client priors; ship bandit anyway with `--bandit-prior` per client |
| `--prefill-keep-ratio` default reinterpretation breaks downstream tooling | Day 6 review | Keep as fixed default; bandit opt-in via `extra_body.session_id` presence (already additive) |

## What success looks like at end of week

- **One PR** on `main`, ~220 LOC, no kernel touches
- **Default API contract**: client sends `/v1/messages` with no `keep_ratio` and no `pflash_mode`. Server self-tunes per session from DFlash chain accept_rate. Quality preserved (claude_code multi-turn 3/3 OK_DONE). No regression vs the static-keep=0.05 baseline.
- **Per-session JSONL traces** demonstrating bandit convergence on ≥ 2 of 3 client harnesses
- **README + `--help`** explaining the no-knob behavior

## What we tell mrciffa at ship

> Adaptive keep_ratio bandit landed on `main`. Server self-tunes per session from DFlash chain accept_rate. Client sends nothing — no `keep_ratio`, no `pflash_mode` — and the server picks the right compression for the workload turn-by-turn. Validated on claude_code and opencode multi-turn at 32K. ~220 LOC, one PR, no kernel changes. The skip+anchor work stays separate on the evidence branch as `pflash_mode=always` opt-in for users who explicitly want the prefill speedup. That's the MVP you asked for; the rest is extension material.

## Drift discipline (the lesson from today)

The chronos review confirmed that today's "drift" produced solid bench foundations (envelope, anchor coverage, composition, real-transcript study) but ALSO produced a paper plan, scaling roadmap, v2 ideas, and Momus/Codex critiques that are **text-only without experiments backing them**. This PLAN.md retains all of those as future work but **does not let them block the ship**. The bandit is the ship; everything else is a follow-up.

If anyone — including me — proposes adding scope to this PR, the answer is "make it a follow-up PR." No exceptions.
