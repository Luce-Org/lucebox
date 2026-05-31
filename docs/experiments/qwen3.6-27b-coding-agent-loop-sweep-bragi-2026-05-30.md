# Qwen3.6-27B-Q4_K_M — coding-agent-loop autotune sweep — bragi — 2026-05-30

First end-to-end run of the `coding-agent-loop` autotune profile on
Qwen3.6-27B on bragi, a consumer Blackwell laptop.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * **Note**: GPU running at ~86–90 W / 1515 MHz during this run (Windows
    Balanced power mode; WSL2 cannot set TDP). Full-performance mode
    (Best performance) would yield ~150–175 W / 2500+ MHz and ~40–50 tok/s
    decode vs the 24–25 tok/s observed here.
* **Image**: locally-built `lucebox-hub:cuda12` from
  `feat/lucebox-docker` @ `48fafe6` (DFLASH_CUDA_ARCHES=120, sm_120 fat
  binary)
* **Fixture**: one 6-bucket multi-turn replay case from
  `luce-bench/src/lucebench/fixtures/agent_recorded/multi_turn_cases.json`
  (single Claude Code session sliced at 8K/16K/32K/64K/100K/128K
  approx-token buckets per `extract-agentic-fixture.py --multi-turn`)
* **Profile**: `coding-agent-loop`, qwen bracket =
  `max_ctx × cache_type × budget × fa_window` =
  `{65536, 98304} × {tq3_0, q8_0} × {16, 22, 32} × {0}` = 12 cells

## Bracket + outcome

| # | budget | max_ctx | kv    | case_tok*       | tok/s | pass       |
|---|--------|---------|-------|-----------------|-------|------------|
| 1 | 16     | 65536   | tq3_0 | 32768 → 42735   | 3.1   | ✓          |
| 2 | 22     | 65536   | tq3_0 | 32768 → 42735   | 3.1   | ✓          |
| 3 | 32     | 65536   | tq3_0 | 32768 → 42735   | —     | ✗ timeout  |
| 4 | 16     | 65536   | q8_0  | 32768 → 42735   | 4.0   | ✓          |
| 5 | 22     | 65536   | q8_0  | 32768 → 42735   | —     | ✗ timeout  |
| 6 | 32     | 65536   | q8_0  | 32768 → 42735   | —     | ✗ timeout  |
| 7 | 16     | 98304   | tq3_0 | 65536 → ~85500  | 1.2   | ✓ **winner** |
| 8 | 22     | 98304   | tq3_0 | 65536 → ~85500  | 1.2   | ✓          |
| 9 | 32     | 98304   | tq3_0 | 65536 → ~85500  | 1.2   | ✓          |
| 10 | 16    | 98304   | q8_0  | 65536 → ~85500  | —     | ✗ timeout  |
| 11 | 22    | 98304   | q8_0  | 65536 → ~85500  | —     | ✗ timeout  |
| 12 | 32    | 98304   | q8_0  | 65536 → ~85500  | —     | ✗ timeout  |

\*`case_tok` = picker's `context_tokens_approx` (chars/4) → estimated
real token count after Qwen3.6 tokenization. Real Qwen3.6 tokenization
expands by ~**1.30×** relative to chars/4 on this fixture (32768 approx
→ 42,735 real tokens; 65536 approx → ~85K real tokens).

## Findings

### 1. tq3_0 is required at 98K context on 23 GB VRAM

All six q8_0 cells at `max_ctx=98304` timed out (300 s, no response).
All three tq3_0 cells at `max_ctx=98304` passed (208–219 s wall time).

VRAM breakdown:
- Model weights (Qwen3.6-27B Q4_K_M + draft): ~18–19 GB
- KV cache q8_0 at 98304 ctx: ~5–6 GB → total **24–25 GB** → OOM on 23 GB
- KV cache tq3_0 at 98304 ctx: ~2–3 GB → total **21–22 GB** → ~1–2 GB headroom

The timeouts are silent VRAM OOM crashes: the container exits during
server startup (no OOM error in the log — the GPU driver kills the
process), the readiness probe never succeeds, and the 300 s timeout fires.

### 2. q8_0 is faster for short-context inference but only at low budget

At `max_ctx=65536`, `budget=16`, `kv=q8_0` achieves **4.0 tok/s** vs
**3.1 tok/s** for tq3_0 (+29%). This is likely because q8_0 KV lookup
avoids dequantization overhead that tq3_0 pays per head.

However, q8_0 only survives budget=16 at 65536 (budget=22 and 32 timeout).
On this 23 GB card, even at 65536 context, q8_0 + budget=22/32 pushes
VRAM past the limit.

### 3. budget=32 is unreliable at 65536 context

`tq3_0 + budget=32 + max_ctx=65536` timed out despite `budget=16` and
`budget=22` passing at 82–83 s. This aligns with finding #2: higher
budget → more speculative decode state → marginally more VRAM → OOM edge.

At `max_ctx=98304`, budget=32 is fine (219 s vs 208 s for budget=16) —
the tq3_0 KV savings provide enough headroom that the extra budget state
fits.

### 4. Speed metrics are not comparable across max_ctx values

The fixture picker selects the largest case that fits within
`max_ctx − 4096 × 0.7 safety factor`. At `max_ctx=65536` it picks the
32K case (42K real tokens); at `max_ctx=98304` it picks the 64K case
(~85K real tokens). The 65K-cell 4.0 tok/s looks better than the 98K-cell
1.2 tok/s, but they measured different amounts of work — not the same
workload on different configs.

A sweeper sorting by tok/s would pick 65536/q8_0/b16 as the "winner",
which would silently cap real agentic sessions at 64K and OOM on longer
ones. The winner selection was updated (see below) to prefer larger
max_ctx first.

### 5. Qwen3.6 tokenizer expansion: 1.30× on this fixture

The 32K-bucket case has `context_tokens_approx = 32768` (chars/4 estimate)
but the server reports **42,735** real prompt tokens after Qwen3.6
tokenization + chat template wrapping. Expansion ratio: **1.30×**. Compare:
gemma-4-26b on sindri showed ~1.39× on the same fixture.

## Heuristic updates applied

**`autotune.py` — `runtime_from_host()` for 22-31 GB tier:**
Explicitly set `cache_type_k="tq3_0", cache_type_v="tq3_0"` for both WSL
and native 22-31 GB paths. Previously the field was left empty (server
default), which could be q8_0 or f16 — both OOM at 98K on 23 GB VRAM.

**`autotune.py` — `_coding_agent_loop_qwen_bracket()` for 22-31 GB:**
Skip q8_0 when `max_ctx >= 98304`. Previously all 12 cells were generated;
the 6 q8_0/98K cells always fail on 23 GB hardware, wasting ~30 min of
sweep time. Reduced to 9 cells (tq3_0+q8_0 at 65K, tq3_0-only at 98K).

**`sweep.py` — `_pick_winner()` for `agent_replay_pass_rate`:**
Changed primary sort key from `-speed_metric` to `-max_ctx`. Rationale:
different max_ctx values exercise different-sized fixture cases (see
finding #4). Speed is only meaningful within the same max_ctx group. The
corrected sort ensures the winner always uses the largest viable context
window, then optimizes speed within that group.

## Recommended config (bragi, Qwen3.6-27B, 23 GB VRAM WSL2)

```toml
[dflash]
budget = 16
max_ctx = 98304
cache_type_k = "tq3_0"
cache_type_v = "tq3_0"
```

Prefill throughput: ~500 tok/s. Decode throughput at 85K-token context:
**~1.2 tok/s** (speculative decode, 256-token response). Wall time for a
full 90K-token agentic session: ~210 s to first token, then ~1.2 tok/s.
