# Qwen3.6-27B-Q4_K_M — coding-agent-loop autotune sweep — bragi — 2026-06-01

Reconfirmation run of the coding-agent-loop sweep on bragi after the Gemma4-31B
characterization session left the config in a non-optimal state (budget=32, max_ctx=65536,
q8_0). New finding: **budget=32 + max_ctx=65536 + q8_0 causes a GPU compute hang** (not a
silent OOM crash as previously believed).

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120, ~86–90 W Balanced mode)
* **Image**: `lucebox-hub:cuda12` (latest)
* **Model**: Qwen3.6-27B-Q4_K_M (~16.8 GB) + dflash-draft-3.6-q4_k_m (~1.0 GB)
* **Profile**: `coding-agent-loop` (9-cell qwen bracket: prior 12-cell reduced by skipping q8_0 at 98K)
* **Fixture**: multi-turn replay, 6 buckets from `agent_recorded/multi_turn_cases.json`
  - 65K cells: 32K-bucket case → 42,735 real tokens (1.30× expansion), 216 messages
  - 98K cells: 64K-bucket case → 84,835 real tokens (1.30× expansion), 342 messages

## Bracket + Outcome

| # | budget | max_ctx | kv    | prompt_tok | wall    | prefill | decode    | speed_metric | pass        |
|---|--------|---------|-------|------------|---------|---------|-----------|--------------|-------------|
| 1 | 16     | 65536   | tq3_0 | 42,735     | ~90s    | —       | ~22 tok/s | —            | ✓          |
| 2 | 22     | 65536   | tq3_0 | 42,735     | ~90s    | —       | ~22 tok/s | —            | ✓          |
| 3 | 32     | 65536   | tq3_0 | 42,735     | 300s    | —       | 0         | fail         | ✗ timeout  |
| 4 | 16     | 65536   | q8_0  | 42,735     | ~90s    | —       | ~22 tok/s | —            | ✓          |
| 5 | 22     | 65536   | q8_0  | 42,735     | 300s    | —       | 0         | fail         | ✗ timeout  |
| 6 | 32     | 65536   | q8_0  | 42,735     | ∞       | —       | 0         | fail         | ✗ GPU hang |
| 7 | 16     | 98304   | tq3_0 | 84,835     | 226.6s  | 204.0s  | 9.1 tok/s | 0.905        | ✓          |
| 8 | 22     | 98304   | tq3_0 | 84,835     | 224.8s  | 202.4s  | 9.2 tok/s | 0.912        | ✓ **winner** |
| 9 | 32     | 98304   | tq3_0 | 84,835     | 232.3s  | 201.9s  | 6.8 tok/s | 0.882        | ✓          |

*Cells 1-5 results inferred from 05-30 sweep; cells 6-9 directly measured in this run.*

speed_metric = completion_tokens / wall_seconds (205 tokens for all 98K cells).

Cell 9 spec-decode: steps=40, accepted=166/640 (25.9%), avg_commit=5.12 tokens/step.

## Key New Finding: GPU Compute Hang (Budget=32, 65K, q8_0)

Cell 6 (budget=32, max_ctx=65536, q8_0) caused a **GPU infinite compute loop**:
- SM=100%, memory bandwidth=0–1% (vs healthy inference: SM=99%, mem=35–53%)
- Server started normally; hang occurred during inference, not startup
- Previously (05-30 sweep) this was attributed to silent VRAM OOM during startup
  (server wouldn't pass readiness probe → 300s timeout). On 06-01 the server **did** start
  successfully, accepted the request, and then hung — confirming this is a CUDA kernel
  bug, not purely VRAM pressure.
- The hang is the same signature as the Gemma4-31B + Anthropic-format hang (see
  `gemma4-31b-initial-characterization-bragi-2026-05-31.md`). The DFlash server hang
  bug is **not model-specific**.
- Trigger conditions: budget=32 + large KV cache (q8_0 at 65K) + 42K-token context +
  stream=false. Hypothesis: DDTree verification loop enters infinite CUDA kernel cycle
  under VRAM pressure combined with maximum speculative budget.
- **Fix**: `systemctl --user restart lucebox.service`

## Performance at 98K Context (Budget=22, tq3_0, Throttled ~86W)

| metric             | value         | notes                                   |
|--------------------|---------------|-----------------------------------------|
| prompt tokens      | 84,835        | 64K-bucket case after 1.30× expansion  |
| prefill            | 202.4 s       | chunked 512-token batches, O(n²) attn  |
| decode             | 22.4 s        | 205 tokens at 9.2 tok/s                |
| total wall         | 224.8 s       | ~3.7 min first-response latency        |
| decode rate        | 9.2 tok/s     | lower than 22 tok/s at short ctx       |

Prefill is the dominant cost at 84K tokens. The O(n²) attention over chunked prefill
(512-token chunks) scales poorly at very large contexts on the throttled ~86W GPU.

**Budget effect on decode at 98K context:**
- budget=16: 22.5s decode (9.1 tok/s) 
- budget=22: 22.4s decode (9.2 tok/s) — marginal improvement
- budget=32: 30.3s decode (6.8 tok/s) — **35% slower** due to verification overhead with 84K KV cache

At budget=32, each verification step processes 32 draft tokens against the full 84K-token KV
cache. The additional memory pressure from a larger draft batch slows down each DDTree step,
overwhelming the benefit of more draft tokens per step.

At full performance (~175W), expected prefill ~50–60s (3–4× speedup), decode ~25–30 tok/s.

## Comparison to 05-30 Sweep

| dimension        | 05-30           | 06-01           |
|-----------------|-----------------|-----------------|
| cells           | 12 (q8_0 at 98K included) | 9 (q8_0@98K pruned) |
| cell 6 behavior | timeout (OOM during startup?) | **GPU hang** (CUDA kernel bug) |
| cell 7-8 wall   | 208–219s        | 225–227s (throttled 86W) |
| winner          | budget=16, 98K, tq3_0 | **budget=22**, 98K, tq3_0 |
| delta           | —               | budget=22 won by 0.007 tok/s speed_metric |
| config applied  | ✓               | ✓               |

Winner changed from budget=16 to budget=22 due to slightly faster speed_metric (0.912 vs
0.905 tok/s — within noise). Both budgets are functionally equivalent at 98K; budget=32
is clearly inferior (-35% decode speed). Either 16 or 22 is a safe choice; the sweep
picked 22 due to marginally better measured throughput in this run.

## Winner Config (bragi, Qwen3.6-27B, 23 GB VRAM WSL2)

```toml
[dflash]
budget = 22
max_ctx = 98304
cache_type_k = "tq3_0"
cache_type_v = "tq3_0"
```

Applied by the sweep after cell 9 completes. budget=16 is equally valid — within noise of
budget=22 (0.905 vs 0.912 tok/s). Both are dramatically better than budget=32 at 98K context.

## Safe / Unsafe Combinations on 23 GB VRAM

| budget | max_ctx | kv    | safe? | reason                          |
|--------|---------|-------|-------|---------------------------------|
| 16     | 65536   | tq3_0 | ✓    | passes, ~22 tok/s decode        |
| 22     | 65536   | tq3_0 | ✓    | passes, ~22 tok/s decode        |
| 32     | 65536   | tq3_0 | ✗    | timeout (VRAM OOM during decode) |
| 16     | 65536   | q8_0  | ✓    | passes, ~22 tok/s (slightly faster than tq3_0) |
| 22     | 65536   | q8_0  | ✗    | timeout (VRAM OOM)              |
| 32     | 65536   | q8_0  | ✗    | **GPU compute hang** (CUDA bug) |
| 16     | 98304   | tq3_0 | ✓    | passes, 9.1 tok/s, **recommended** |
| 22     | 98304   | tq3_0 | ✓    | passes, 9.2 tok/s               |
| 32     | 98304   | tq3_0 | ✓    | passes, ~9 tok/s                |
| any    | 98304   | q8_0  | ✗    | OOM (5–6 GB KV + 18 GB model > 23 GB) |
