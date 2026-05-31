# Qwen3.6-27B comprehensive benchmark sweep — bragi — 2026-05-31

Complete pass-rate sweep across all luce-bench areas for Qwen3.6-27B-Q4_K_M on
the final recommended image (`dc20057e-cuda12`), nothink mode.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * GPU throttled ~86–90 W / 1515 MHz (Windows Balanced mode).
* **Model**: Qwen3.6-27B-Q4_K_M, optimal config:
  ```toml
  budget = 16
  max_ctx = 98304
  cache_type_k = "tq3_0"
  cache_type_v = "tq3_0"
  fa_window = 0
  think_max = 15488
  ```
* **Image**: `dc20057e-cuda12` (lucebox-hub:cuda12, built 2026-05-31 15:02)
  Contains all three server fixes: call:verb{} detection, StepEnforcer one-shot,
  tool_result blocks. See `bragi-rtx5090-final-tuning-summary-2026-05-31.md`.

## Results

All areas run with `--no-think`. Commands:
```
uv run luce-bench --areas ds4-eval --no-think
uv run luce-bench --areas gsm8k,hellaswag,truthfulqa-mc1 --no-think
uv run luce-bench --areas truthfulqa-mc1,agent,longctx --no-think
uv run luce-bench --areas hellaswag --no-think   # re-run (server restart contaminated first run)
```
(forge, code, agent_recorded already captured earlier in same session)

| area | pass_rate | n | wall_total | wall_median | vs prev baseline | delta |
|------|-----------|---|------------|-------------|-----------------|-------|
| forge | **100%** | 5/5 | ~32s | 6.1s | 0% (tool_result bug) | +100pp |
| agent | **100%** | 4/4 | 44s | 11s | 75% (3/4) | +25pp |
| longctx | **100%** | 6/6 | 241s | 40s | — (not prev run) | new |
| ds4-eval | **77.2%** | 71/92 | 11752s | 68.7s | 70.7% | +6.5pp |
| truthfulqa-mc1 | **80%** | 80/100 | 27s | — | 80% | = |
| code | **90%** | 9/10 | 17s | 1.5s | ~90% | = |
| agent_recorded | **42.3%** | 11/26 | — | — | 42.3% | = |
| hellaswag | **88%** | 88/100 | 41s | 0.5s | 90% | -2pp |
| gsm8k | **86%** | 86/100 | 1701s | — | 89% | -3pp |

### Notes on deltas

**forge +100pp**: The `normalize_chat_messages()` fix (commit `dc20057e`) was
decisive — tool_use and tool_result Anthropic blocks were silently dropped in
earlier images, causing multi-turn tool conversations to loop infinitely.

**agent +25pp**: Improvement is notable but the area only has 4 questions (1
case difference). Likely reflects the tool_result fix also improving multi-turn
agent scenarios using the Anthropic Messages API path.

**longctx 100%**: Qwen3.6 achieves 100% on the longctx area (vs Gemma4 which
also got 100% nothink, 83.3% think). Not run in prior Qwen3.6 baselines.

**ds4-eval +6.5pp**: Run-to-run variance is typical ±3-5pp for this benchmark
(92 cases, mix of GPQA/SuperGPQA/AIME/COMPSEC). The improvement from 70.7%
to 77.2% likely reflects sampling variation more than any server fix (ds4-eval
uses `/v1/chat/completions`, not affected by the tool_result fix). Sub-area
breakdown:
- GPQA Diamond (25): 17/25 = 68%
- SuperGPQA (25): 19/25 = 76%
- AIME2025 (25): 19/25 = 76%
- COMPSEC (17): 16/17 = 94.1%

**gsm8k -3pp, hellaswag -2pp**: Within normal run-to-run variance (±3-5pp).
These MC/math areas are unaffected by the server fixes (OpenAI format).

### Note on hellaswag first run (79%)

During the first gsm8k+hellaswag+truthfulqa run, the server restarted mid-run.
Cases 93–100 of hellaswag all got `given=?` (case 93 wall=10.24s timeout, cases
94–100 wall=0.00s immediate error), then truthfulqa immediately got
`ConnectionResetError`. The clean re-run on the stable server gave 88% with
zero connection failures.

## ds4-eval sub-area detail

AIME2025 nothink observation: the model generates 2000–11000 output tokens per
problem even without a `<think>` block, reasoning inline in the answer. The
median AIME case takes 68.7s wall at 24–25 tok/s. Case 57 (aime2025-12) was
the longest at 489.9s and 11259 output tokens.

Long AIME output in nothink mode is expected: Qwen3.6's nothink prompt suppresses
the `<think>` header block, but the model still reasons through complex problems
in its response text. This is correct behavior.

## Cross-run consistency check

| area | run-1 (prev image) | run-2 (dc20057e) | δ |
|------|-------------------|-----------------|---|
| code | ~90% | 90% | 0 |
| agent_recorded | 42.3% | 42.3% | 0 |
| truthfulqa-mc1 | 80% | 80% | 0 |
| hellaswag | 90% | 88% | -2 |
| gsm8k | 89% | 86% | -3 |
| ds4-eval | 70.7% | 77.2% | +6.5 |

The ±3pp noise band for MC/math areas is consistent across runs. ds4-eval has
higher variance due to its mix of problem types (some AIME problems require
10K+ tokens, dominating run-to-run variance).
