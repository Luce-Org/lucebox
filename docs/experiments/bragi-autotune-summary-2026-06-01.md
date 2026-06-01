# Bragi Auto-Tune Summary — 2026-06-01

Complete record of tuning decisions for bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2).
Covers sessions 2026-05-30 through 2026-06-01.

## Hardware

| spec | value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| VRAM | 23 GB (24,463 MiB) |
| Platform | WSL2 (Windows Balanced ~86–90W, 1515 MHz) |
| CPU | Intel Core Ultra 9 275HX (24T) |
| Driver | 596.36 |

**Note:** Running at Windows Balanced power mode → ~86–90W TDP, decode ~22–25 tok/s.
At Best Performance (~175W), expect ~40–50 tok/s decode.

## Optimal Configuration (Qwen3.6-27B, 2026-06-01)

```toml
[dflash]
budget = 22              # empirically best; 16 equivalent, 32 is 35% slower at 98K ctx
max_ctx = 98304          # maximum safe for 23 GB VRAM with tq3_0 KV
cache_type_k = "tq3_0"  # required at 98K (q8_0 would OOM: 5-6 GB KV + 18 GB model > 23 GB)
cache_type_v = "tq3_0"
prefix_cache_slots = 0   # KEEP AT 0 — regression bug at >0 (see below)
prefill_mode = "off"     # pflash needs prefix_cache_slots > 0 to do anything
fa_window = 0            # swept; 0 is optimal when pflash is off

[model]
preset = "qwen3.6-27b"

[autotune]
source = "sweep"
timestamp = "2026-06-01T06:30:52Z"
```

## Tunable Sweep Results

### DFlash Budget

| budget | max_ctx | kv    | result        | notes                           |
|--------|---------|-------|---------------|---------------------------------|
| 16     | 65536   | tq3_0 | ✓ pass        | ~22 tok/s decode                |
| 22     | 65536   | tq3_0 | ✓ pass        | ~22 tok/s decode                |
| 32     | 65536   | tq3_0 | ✗ timeout     | OOM during decode               |
| 16     | 65536   | q8_0  | ✓ pass        | ~22 tok/s (slightly faster KV lookup) |
| 22     | 65536   | q8_0  | ✗ timeout     | OOM                             |
| 32     | 65536   | q8_0  | ✗ **GPU hang** | Infinite CUDA kernel loop      |
| 16     | 98304   | tq3_0 | ✓ pass        | 9.1 tok/s at 84K prompt        |
| 22     | 98304   | tq3_0 | ✓ **winner**  | 9.2 tok/s at 84K prompt        |
| 32     | 98304   | tq3_0 | ✓ pass        | 6.8 tok/s — 35% slower at 98K  |

**Finding**: Budget=32 causes GPU hang at 65K+q8_0 and is 35% slower at 98K+tq3_0.
Budget=22 and 16 are equivalent at 98K; winner selection picked 22 by 0.007 tok/s margin.

### max_ctx

| max_ctx | kv    | model  | viable? | notes                          |
|---------|-------|--------|---------|--------------------------------|
| 65536   | tq3_0 | Qwen3.6 | ✓     | ~22 tok/s decode; fits easily  |
| 98304   | tq3_0 | Qwen3.6 | ✓     | ~9 tok/s at 84K (204s prefill) |
| 98304   | q8_0  | Qwen3.6 | ✗     | OOM (5-6 GB KV + 18 GB model > 23 GB) |

98304 (96K) is the maximum safe context for Qwen3.6-27B on 23 GB VRAM with tq3_0.
Handles agentic sessions up to ~90K tokens (1.30× tokenizer expansion from approx token count).

### KV Quantization

| kv    | at 65K        | at 98K  | VRAM at 98K |
|-------|---------------|---------|-------------|
| f16   | ≈ 2.8 GB      | ≈ 4.4 GB | OOM        |
| q8_0  | ≈ 1.4 GB      | ≈ 2.2 GB | OOM (total ≈ 20.2 GB + overheads) |
| tq3_0 | ≈ 0.6 GB      | ≈ 0.9 GB | Safe (total ≈ 19.7 GB) |

Qwen3.6-27B KV formula: `28 layers × 8 KV heads × 128 head_dim × 2 × bits/8`
= ~21.5 KB/token (tq3_0), ~43 KB/token (q8_0).

### Prefix Cache

| prefix_cache_slots | result | notes                         |
|-------------------|--------|-------------------------------|
| 0                  | ✓      | baseline — use this           |
| 32                 | -19pp  | agent_recorded 23.1% vs 42.3% |

Daemon snapshot path bug causes incorrect KV cache reuse for multi-turn tool calls.
**Keep prefix_cache_slots=0 until the bug is fixed.**
Reference: `docs/experiments/qwen3.6-27b-prefix-cache-regression-bragi-2026-05-31.md`

### PFlash (prefill compression)

PFlash requires `prefix_cache_slots > 0` to compress anything useful. Since the prefix
cache regression precludes using prefix_cache_slots, pflash is effectively disabled.
`prefill_mode = "off"` is the correct setting.

Reference: `docs/experiments/qwen3.6-27b-pflash-ab-test-bragi-2026-05-31.md`

### fa_window

Swept as part of the gemma4-26b coding-agent-loop sweep (2026-05-30). fa_window=0 is
optimal when pflash is off. At fa_window=2048 with pflash, some improvement is possible
but requires prefix_cache_slots to be working.

## Known Issues / Blockers

### 1. DFlash Server Hang Bug

**Trigger**: GPU enters infinite compute loop (SM=100%, mem=0-1%) — observed with:
- Gemma4-31B + Anthropic-format multi-message+tool request (forge client killed mid-run)
- Qwen3.6-27B + budget=32 + max_ctx=65536 + q8_0 + 42K-token context

**Symptoms**: http_status=None, TimeoutError for all cases; `/health` OK but no inference

**Fix**: `systemctl --user restart lucebox.service`

**Mitigation**: Use the sweep-validated safe configs (budget≤22 at 65K, any budget at 98K tq3_0)

### 2. Prefix Cache Regression

-19pp on agent_recorded with prefix_cache_slots=32. Bug in daemon snapshot path for
multi-turn tool calls. Keep at 0.

## Model Matrix (bragi, 2026-06-01)

| model | VRAM | max_ctx | decode | forge | agent | code | gsm8k | notes |
|-------|------|---------|--------|-------|-------|------|-------|-------|
| Qwen3.6-27B | 18 GB | 98K | 22-25 tok/s | 100% | 34.6%† | 90% | 81% | **preferred** |
| Laguna-XS.2 | 22 GB | 32K | 60-63 tok/s | 0%* | 50% | 20%** | 93% | fast math |
| Gemma4-31B | 22 GB | 32K | 22-23 tok/s | 0%* | 38.5% | 70% | 95% | best math |
| Gemma4-26B-A4B | 15 GB | 65K+ | ~25 tok/s | 0%* | 19% | 0%*** | 84% | tool bug |
| Qwen3.6-MoE | 22 GB | 32K | ~70-90 tok/s | n/a | n/a | n/a | n/a | no DFlash |

\*Model can't emit tool_use blocks.  
\*\*FIM format mismatch, not capability gap.  
\*\*\*Token leak bug in old image.  
†Winner-config run (budget=22, 98K, tq3_0): 9/26 (34.6%). Prior dc20057e baseline: 10/26 (38.5%). 7 cases
 flipped (3 FAIL→PASS, 4 PASS→FAIL) — 1-case delta is within noise at n=26 (σ≈9.5pp). No regression.

## Performance Reference (Qwen3.6-27B, budget=22, 98K, tq3_0, throttled 86W)

| context  | prefill  | decode    | notes                         |
|----------|----------|-----------|-------------------------------|
| <5K      | <1s      | 22-25 tok/s | typical agent turn          |
| 42K      | ~5s      | 22 tok/s  | DFlash 65K sweep case         |
| 85K      | 204s     | 9.1 tok/s | DFlash 98K sweep case         |

At full performance (~175W), prefill ~3-4× faster; decode ~40-50 tok/s.

## Sweep History

| date       | profile              | winner                           | key finding                         |
|------------|----------------------|----------------------------------|-------------------------------------|
| 2026-05-30 | coding-agent-loop    | budget=16, 98K, tq3_0           | q8_0 at 98K OOMs; tok/s ≠ quality proxy |
| 2026-06-01 | coding-agent-loop    | budget=22, 98K, tq3_0           | budget=32 hangs at 65K+q8_0; 35% slower at 98K |
| 2026-06-01 | agent_recorded QA    | (winner confirmed)               | 9/26 (34.6%) vs baseline 10/26 (38.5%) — within noise |

Reference experiments:
- `qwen3.6-27b-coding-agent-loop-sweep-bragi-2026-05-30.md`
- `qwen3.6-27b-coding-agent-loop-sweep-bragi-2026-06-01.md`
- `qwen3.6-27b-prefix-cache-regression-bragi-2026-05-31.md`
- `qwen3.6-27b-pflash-ab-test-bragi-2026-05-31.md`
- `gemma4-31b-initial-characterization-bragi-2026-05-31.md`
- `laguna-xs2-initial-characterization-bragi-2026-05-31.md`
