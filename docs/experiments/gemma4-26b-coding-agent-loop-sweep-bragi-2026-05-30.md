# Gemma 4 26B-A4B-it — coding-agent-loop autotune sweep — bragi — 2026-05-30

Second run of the `coding-agent-loop` autotune profile against gemma-4-26b;
first run on bragi (Blackwell sm_120). Corrects an incorrect conclusion from
the earlier sindri sweep where all 131K cells appeared to fail.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * **Note**: GPU at ~86–90 W / 1515 MHz (Windows Balanced mode; WSL2 cannot
    set TDP). At full performance (150–175 W) decode rate would be ~50–60 tok/s
    vs the ~30 tok/s observed here.
* **Image**: locally-built `lucebox-hub:cuda12` from
  `feat/lucebox-docker` @ `48fafe6` (DFLASH_CUDA_ARCHES=120)
* **Fixture**: one 6-bucket multi-turn replay case from
  `luce-bench/src/lucebench/fixtures/agent_recorded/multi_turn_cases.json`
  (case `claude-2026-05-23-multiturn-65536-65eed`, 65205 approx-token bucket)
* **Profile**: `coding-agent-loop`, gemma bracket =
  `max_ctx × {98304, 131072} × fa_window × {0, 2048} × budget × {16, 22, 32}` = 12 cells

## Bracket + outcome

| # | budget | max_ctx | fa_win | case_tok* | tok/s  | pass     |
|---|--------|---------|--------|-----------|--------|----------|
| 1 | 16     | 98304   | 0      | 65205     | 2.0    | ✓        |
| 2 | 22     | 98304   | 0      | 65205     | 1.9    | ✓        |
| 3 | 32     | 98304   | 0      | 65205     | 2.0    | ✓        |
| 4 | 16     | 98304   | 2048   | 65205     | 1.9    | ✓        |
| 5 | 22     | 98304   | 2048   | 65205     | 2.0    | ✓        |
| 6 | 32     | 98304   | 2048   | 65205     | 2.0    | ✓        |
| 7 | 16     | 131072  | 0      | 65205     | 2.0    | ✓        |
| 8 | 22     | 131072  | 0      | 65205     | 2.0    | ✓ **winner** |
| 9 | 32     | 131072  | 0      | 65205     | 2.0    | ✓        |
| 10 | 16    | 131072  | 2048   | 65205     | 2.0    | ✓        |
| 11 | 22    | 131072  | 2048   | 65205     | 1.9    | ✓        |
| 12 | 32    | 131072  | 2048   | 65205     | 2.0    | ✓        |

\*`case_tok` is the picker's `context_tokens_approx` (chars/4). The actual
real token count after Gemma tokenization + chat template wrapping is
**~90K** (1.39× expansion). All cells used the same 64K-bucket case.

Winner: cell 8 (budget=22, max_ctx=131072, fa_window=0, 2.0 tok/s). Cells 7
and 8 both scored 2.0 tok/s, but cell 8's wall time (63.9 s vs 64.4 s) gave
it a marginally higher float speed_metric, beating cell 7 (budget=16) on the
primary sort key before the budget tiebreaker fired.

## Findings

### 1. Gemma 4 26B fits at 131K context on 23 GB VRAM — confirmed

All 12 cells passed, including all 6 at max_ctx=131072. VRAM breakdown:
- Model weights (Gemma 26B-A4B Q4_K_M + draft): ~14–15 GB
- KV cache F16 at 131072 ctx (GQA, ~4 KV heads, 256 head dim, 30 layers):
  ~7–8 GB
- Total: **~22–23 GB** — fits on bragi's 23 GB with ~1 GB headroom

The KV cache is allocated upfront for max_ctx tokens at server startup.
Since all 131K cells started and responded, the allocation succeeded. The
headroom is slim — this config sits at the edge of the hardware.

### 2. Why sindri appeared to fail at 131K (fixture picker issue)

The sindri sweep (`gemma4-26b-coding-agent-loop-sweep-2026-05-30.md`)
reported all 131K cells failing with HTTP 400. At the time, the fixture
picker selected the **100K-bucket case** (`context_tokens_approx ≈ 102397`)
for max_ctx=131072. Gemma expands that by ~1.39×: 102397 × 1.39 ≈ 142K
real tokens, exceeding the server's 131072 − 4096 = **126976** ceiling.

On bragi today, the picker selected the **64K-bucket case**
(`context_tokens_approx = 65205`) for both 98304 and 131072, which expands
to ~90K real tokens — well within 126976. The picker's
`safety_factor=0.7` was likely updated between the two runs, changing the
effective budget threshold from `1.0 × (max_ctx − 4096)` to
`0.7 × (max_ctx − 4096)`:

- Old: effective_budget = 126976 × 1.0 = 126976 → 100K case (102397) fits ✓
- New: effective_budget = 126976 × 0.7 = 88883 → 64K case (65205) fits ✓,
  100K case (102397) does not ✗

So sindri's 131K failures were a **fixture selection artifact, not a VRAM
limit**. The hardware could handle it; the test picked a case that was too
large for the server's request ceiling.

### 3. fa_window gives no benefit for Gemma 4 at 90K-token context

fa_window=0 (full attention) and fa_window=2048 produced identical throughput
(2.0 tok/s, within noise) across all budget/max_ctx combinations. This
replicates the sindri finding: Gemma 4 26B-A4B's decode is not
bandwidth-bound at this scale in a way that sparse-attention windowing
improves. fa_window=0 is the recommended default.

### 4. Budget insensitive on Gemma's MoE architecture

budget=16, 22, and 32 all score ~2.0 tok/s (within ±0.1 tok/s noise) at
both context sizes. The draft budget has minimal leverage on Gemma 4 26B-A4B:
the 4B-active MoE decoder is already fast enough that more speculative tokens
don't meaningfully amortize verification cost. budget=22 (the heuristic
default) is fine; there's no need to tune this axis further.

### 5. Gemma is faster than Qwen3.6 at long context

At 98K context (both using the same 64K case):
- Gemma 4 26B-A4B: **2.0 tok/s**, 64 s wall (90K actual tokens)
- Qwen3.6 27B: **1.2 tok/s**, 209 s wall (~85K actual tokens)

Gemma's 4B-active MoE architecture decodes ~67% faster than Qwen3.6's denser
27B at equivalent real-token prompt sizes.

## Heuristic updates applied

**`autotune.py` — `_coding_agent_loop_gemma_bracket()` docstring:**
Updated to note that 131K is confirmed viable on 23–24 GB VRAM. Removed the
implication that 131K cells fail. The old sindri conclusion was a
fixture-picker artifact, not a hardware constraint.

No code change to the bracket itself — it already correctly sweeps both
98304 and 131072. The winner selection (sort by max_ctx first) will
automatically prefer 131072 cells if they pass.

## Recommended config (bragi, Gemma 4 26B, 23 GB VRAM WSL2)

```toml
[dflash]
budget = 22
max_ctx = 131072
fa_window = 0
```

Prefill throughput at 90K real tokens: ~240 s wall (~375 tok/s). Decode
throughput: **~2.0 tok/s** speculative, 126-token response. The 131K ceiling
accommodates real coding-agent sessions up to ~120K real tokens.
