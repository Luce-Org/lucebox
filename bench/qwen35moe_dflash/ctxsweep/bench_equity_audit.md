# Benchmark Equity Audit: Qwen3.6 DENSE 27B vs MoE 35B-A3B

Date: 2026-06-22. Scope: did the dense 27B get its best shot vs the MoE; is decode the real ceiling or under-tuned; what real agentic-coding bench should re-measure both. Hardware: RTX 3090 (24.6 GiB).

## Part 1 — Arms-Run Matrix (was 27B given a fair shot? NO)

| Config dimension | MoE 35B-A3B arms tried | Dense 27B arms tried | 27B fairness gap |
|---|---|---|---|
| KV cache type | f16, q4_0, q8_0, tq3_0 (full sweep, `ctxsweep/kvsweep.txt`) | q4_0 (✓), f16 (TRUNCATED — `agentic_27b_f16.log` stops mid-turn-1), tq3_0 (confounded/poor) | **MISSING: completed f16-KV agentic run** |
| Best agentic arm KV | f16 + kvflash (66.9 tok/s) | q4_0 (14.0 tok/s) | **NOT like-for-like — headline gap confounded by KV format** |
| max_ctx | 40960 / 65536 / 131072 | 40960 only | MISSING: 64K/128K for 27B |
| kvflash on/off | both (auto breaks prefix-cache, net-neg) | OFF only | MISSING (likely irrelevant — kvflash net-neg) |
| Drafter | dflash-draft-3.6-bf16-reconv (SWA-5, win=2048) | same, only this one | parity OK |
| CUDA graphs | default-on (AR path); no explicit OFF arm | default-on; no explicit OFF arm | parity; neither isolates the lever |
| All-hot vs offload | all-hot only (all experts GPU-resident @40K) | all-hot only (dense fits) | parity |
| MTP | none | **models exist (mtp-q4/q8, froggeric) but ZERO decode benchmarks** | **MISSING: 27B MTP is the dense-specific accelerator, never measured** |
| Context sweep | ctx 1K–32K verbatim + needle | ctx 8K/32K + needle (`dense27b_rebaseline.md`) | parity on synthetic |

### Headline numbers (provenance)
- MoE best agentic: **66.9 tok/s mean** decode, 34.8s wall, 34K ctx, f16 KV — `bench/abc_cache_harness/results/BENCH_35B_20260622_084452_smoke.md`.
- Dense best agentic: **14.0 tok/s mean** decode, 109.6s wall, 34K ctx, q4_0 KV — `bench/abc_cache_harness/results/BENCH_27B_20260622_100403_smoke.md`.
- The dense-vs-MoE doc ITSELF flags this: "The dense vs MoE decode gap (14 vs 67 tok/s) is partly architecture (A3B active-param MoE) and partly KV format. NOT like-for-like on KV." — `ctxsweep/agentic_bestconfig_dense_vs_moe.md`.

### Verdict Part 1
The 4.76× gap is REAL in direction (MoE activates ~3B of 35B params per token; dense runs all 27B) but the MAGNITUDE is inflated by a KV-format confound (MoE=f16, dense=q4_0) and by the dense MTP accelerator never being benched. Three runs are MISSING for a fair 27B comparison:
1. **Dense 27B agentic, f16 KV, 3-turn goldgate replay** (the existing f16 log is truncated). Same arm shape as the MoE best arm.
2. **Dense 27B MTP decode** — model files exist (`~/models/qwen3.6-27b-mtp-q4/q8/froggeric`), zero benchmarks. MTP is the dense-only decode lever the MoE doesn't need.
3. **Dense 27B q8_0 KV** — per PR#372, q8_0 is ~2× faster than tq3_0 at long ctx and avoids the tq3 dequant tax; never run for dense agentic.

## Part 2 — Decode Ceiling Verdict: UNDER-TUNED, not hardware ceiling

### Is 14 tok/s (dense) / 67 tok/s (MoE) the ceiling? NO.

**Dense 27B-Q4 measured ceiling is ~41 tok/s @71K, ~60 tok/s @4K** (memory-bandwidth roofline, nsys-measured) — `project_decode_ceiling_reality.md`. The agentic 14 tok/s is far below this because:
- The long-ctx wall is **kernel-launch overhead (~42 ms/token @71K, CPU-bound, 2020 launches/token), NOT FlashAttention** (FA = 1.3% of compute). Source: `project_decode_ceiling_reality.md`, nsys profile.
- Agentic dense decode is further depressed by gate-floored AR on novel content (spec engages only 1/3 turns) — that is content-dependent, not hardware.

**MoE 67 tok/s** is also below its roofline: AR-decode roofline for 35B-A3B ≈ 541 tok/s; measured ~115 tok/s AR ≈ 21% of roofline — `project_ar_decode_roofline_cudagraphs.md`. Dense all-hot roofline @26K ≈ 600 tok/s vs measured 66 — `project_qwen35moe_decode_roofline_measured.md`.

### tq3_0 KV decode tax IS present and is a known suppressor
- **tq3_0 @63K = 14.4 tok/s vs f16/q8 = 29.3/29.4** (llama-bench, like-for-like) — `project_decode_gap_tq3_kv_dominant.md`. Swapping tq3_0→q8_0 KV is a **~2.0× decode win, env-only**.
- The MoE ctxsweep KV sweep confirms: f16/q4_0 optimal, tq3_0 ~18% slower on decode — `ctxsweep/kvsweep.txt`.

### Ranked un-applied decode levers (with provenance)

| Rank | Lever | Expected gain | Applies to | Provenance | Status |
|---|---|---|---|---|---|
| 1 | **q8_0 KV instead of tq3_0** (long ctx) | ~2.0× @63K | both, dense agentic especially | `project_pr372_shipped_q8_story.md`, `project_decode_gap_tq3_kv_dominant.md` | SHIPPED recipe, not applied to dense agentic arm. Build `DFLASH27B_FA_ALL_QUANTS=OFF` or OOMs |
| 2 | **AR-cudagraph** (already shipped, gated n_tokens==1) | 1.56–1.65× on AR-floored turns | both, dense agentic (AR-heavy) | `project_ar_decode_roofline_cudagraphs.md` (65→107 tok/s) | SHIPPED 2026-06-02. Confirm it's ON in the agentic binary; dense agentic is mostly AR so this is high-value |
| 3 | **f16 KV for dense agentic** | accept parity + decode (vs q4_0/tq3) | dense 27B | `ctxsweep/kvsweep.txt` (f16 fastest) | the MISSING truncated run |
| 4 | **MTP (dense only)** | ~1.1–1.5× IF kernel fixed; ~22–32 tok/s realistic | dense 27B | `project_decode_ceiling_reality.md` | BLOCKED: serial delta-net kernel + reinit crash (`project_mtp_pflash_reinit_bug.md`). Not ship-ready |

DEAD levers (do not pursue): CUDA-graph spec-VERIFY (net −25% @24K, `project_cudagraph_spec_verify_validated.md`); on-GPU argmax on dense (already done, 0.05% of decode, `project_decode_step_breakdown.md`); monokernel (parked, doesn't transfer to 27B-Q4, `project_decode_monokernel_plan.md`).

### Verdict Part 2
14 tok/s dense agentic is UNDER-TUNED by ~2-3 stacked, already-known levers: it runs on q4_0/tq3-class KV with AR-cudagraph possibly not confirmed, and never with MTP. A fair dense re-measure on q8_0 KV + confirmed AR-cudagraph should land materially above 14 — closer to the 26 tok/s already seen cold @32K (`dense27b_rebaseline.md`) and toward the ~41 tok/s roofline. The dense-vs-MoE gap will narrow but not close (MoE's active-param advantage is architectural and real).

## Part 3 — Real Agentic-Coding Bench: use goldgate, NOT synthetic

### What was actually used in the comparison: SYNTHETIC dominates
- ctxsweep `ctx_*.json` = verbatim-copy of a .cpp function (deterministic, ~90% accept) — INFLATES spec-decode.
- needle `*.json` = NIAH recall — synthetic, off-distribution for coding.
- charbench `01-08.json` = 8 single-turn entropy probes (verbatim→prose), not multi-turn.
- bench_agent.jsonl (`harness/benchmarks/prompts/`) = single-turn context-dumps (Django/sympy/requests bug snippets), NOT live multi-turn tool-use.

The content-dependence is the trap: accept is **76-93% on verbatim-copy vs 15-33% on novel prose/code** (`charbench/NOTES.md`, memory `feedback_degenerate_bench_prompts.md`). Synthetic copy prompts inflate dFlash accept ~1.6-2× over real coding. Any 67-vs-14 headline drawn off verbatim/copy regimes is optimistic for both models and especially for the spec-decode arms.

### The REAL agentic-coding asset: `bench/abc_cache_harness/traces/goldgate_fix.jsonl`
EXACT sizing (measured, `wc`):
- **7 turns**, 1,524,085 bytes total. Per-turn bytes: 141K, 145K, 155K, 265K, 266K, 268K, 269K (jump at turn 4 = large tool-output/file-read).
- Schema = real Anthropic messages API: `model, system, tools[], messages[], max_tokens, temperature, stream`. Message counts grow 1→3→5→9→11→13→14 (growing-prefix multi-turn, ends on assistant).
- Task (turn 1): *"The pytest suite for the stats_pkg module is failing. Please investigate and fix all failing tests. The package is at bench/2026-06-06_drafter_ab/goldgate/fixture/stats_pkg/"* — a REAL multi-turn debug-and-fix coding session with embedded tool calls. This is the canonical fair bench.

**Caveat (a real finding):** turn 7 payload ≈ 269K bytes ≈ 68-77K tokens, which EXCEEDS the harness `MAX_CTX = 40960` (`replay_harness.py:55`). The existing 3-turn replays stay under cap (~34-38K tok); a full 7-turn replay needs either a raised cap (and the 27B q8_0 build to fit KV) or truncation. State which when reporting.

The live harness (`harness/clients/run_claude_code.sh` + `repo_inspection.txt`) drives the real claude-code binary but `repo_inspection.txt` is a lightweight 3-question inspection probe, not a code-edit task — use it for tool-call smoke, not for the decode headline.

### Recommended fair re-measurement (both models)

Prompt set: `bench/abc_cache_harness/traces/goldgate_fix.jsonl`, turns 1-3 (stays under 40960 cap; matches existing replays for continuity). Optionally turns 1-7 with raised cap + q8_0 build.
Harness: `bench/abc_cache_harness/replay_harness.py` (port 19099, not 18099 — never contend the user's live server).
Report per turn + mean: prompt_tokens, prefill_s, prefill_TPS (on original prompt), decode_TPS (from `decode=` field), accept τ, wall_s, cache_hit. Baseline arm = dFlash-alone (no pflash). temp=0, fixed seed, binary md5.

Arms to run for a FAIR comparison (the MISSING runs):

| # | Model | KV | Drafter | Graphs | Purpose |
|---|---|---|---|---|---|
| A | Dense 27B | **f16** | dflash-bf16 | AR-cudagraph ON | like-for-like vs MoE-f16 (closes the KV confound) |
| B | Dense 27B | **q8_0** | dflash-bf16 | AR-cudagraph ON | the ~2× long-ctx lever; build FA_ALL_QUANTS=OFF |
| C | MoE 35B | f16 | dflash-bf16 | AR-cudagraph ON | existing best arm, re-run same harness/seed |
| D | Dense 27B | q8_0 | **MTP** | n/a | dense-only accelerator — MISSING, blocked on reinit/kernel; run only after that fix |

Expected outcome: A and B narrow the dense-vs-MoE gap from 4.76× toward ~3× (KV-confound removed); D is the only path to push dense decode toward its ~41 tok/s roofline but is not ship-ready. The architectural MoE advantage (active-param sparsity) survives all of these — it is real, just smaller than 4.76× once the bench is fair.

## MISSING runs summary (name the run needed)
1. **Dense 27B f16-KV 3-turn goldgate** — existing log truncated mid-turn-1. RUN: arm A above.
2. **Dense 27B q8_0-KV agentic** — never run; ~2× lever unused. RUN: arm B (build `DFLASH27B_FA_ALL_QUANTS=OFF`).
3. **Dense 27B MTP decode** — models exist, zero benchmarks; blocked on reinit crash + serial delta-net kernel. RUN: arm D after kernel fix.
4. **Confirm AR-cudagraph is ON** in the benched agentic binary (shipped 2026-06-02, gated n_tokens==1) — dense agentic is AR-heavy so this is load-bearing; verify via binary, not assumption.
5. **Full 7-turn goldgate** (turns 4-7 exceed 40960 cap) — needs raised max_ctx + 27B q8_0 build to fit KV.
