# Bragi RTX 5090 Laptop — Final Tuning Summary — 2026-05-31

Complete picture of autotune results, model selection, and server bug fixes for
bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120).

## Hardware

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 23 GB VRAM (sm_120 / Blackwell)
- CPU: Intel Core Ultra 9 275HX, 24 cores
- RAM: 31 GB
- OS: Ubuntu 24.04 WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2)
- Note: GPU throttled ~86–90 W / 1515 MHz in Windows Balanced power mode.
  All benchmarks captured in this throttled state. Switch to "Best Performance"
  mode for ~40-60% higher throughput in production.

## Recommended Configuration: Qwen3.6-27B

```toml
# ~/.lucebox/config.toml
[model]
preset = "qwen3.6-27b"

[dflash]
budget = 16
max_ctx = 98304
cache_type_k = "tq3_0"
cache_type_v = "tq3_0"
fa_window = 0
think_max = 15488
prefix_cache_slots = 0
prefill_cache_slots = 0
```

**Model**: Qwen3.6-27B-Q4_K_M (17.9 GB on disk)  
**Draft**: dflash-draft-3.6-q4_k_m.gguf  
**Decode speed**: ~24 tok/s (dense 27B active params)  
**Context**: 98304 tokens (128K model max, capped for VRAM)  
**KV quant**: tq3_0 (required to fit 98K context in 23 GB alongside 17 GB model)

## Model Comparison: Qwen3.6-27B vs Gemma4-26B-A4B

All results on image `dc20057e-cuda12` (commit `dc20057e`), bragi throttled state.

| Benchmark | Qwen3.6 nothink | Qwen3.6 think | Gemma4 nothink | Gemma4 think |
|-----------|-----------------|---------------|----------------|--------------|
| **forge** | **100% (5/5)** ★ | — | 20% (1/5) | — |
| **agent_recorded** | 42.3% (11/26) | **46.2% (12/26)** | 19.2% (5/26) | — |
| agent | 75% | 50% | 25% | 50% |
| code | ~90% (9/10) | ~20% | 0% | 0% |
| ds4-eval | 70.7% | **81.5%** | 77.2% | 81.5% |
| gsm8k | **89%** | 82% | 91% | 91% |
| hellaswag | **90%** | 73% | 73% | 42% |
| truthfulqa-mc1 | 80% | **84%** | 77% | 68% |
| longctx | — | — | **100%** | 83.3% |

**Verdict**: Qwen3.6 is the preferred model for coding/agent tasks on bragi.
- forge: Qwen3.6 100% vs Gemma4 20% (+80pp)
- agent_recorded: Qwen3.6 46.2% vs Gemma4 19.2% (+27pp)
- agent: Qwen3.6 75% vs Gemma4 25% (+50pp nothink)
- code: Qwen3.6 ~90% vs Gemma4 0%

Gemma4 advantages (pure reasoning, no tool use): gsm8k (+2pp), longctx (100%),
faster decode speed (66 tok/s vs 24 tok/s due to sparse MoE 4B active params).

## Qwen3.6: think vs nothink guidance

| Use case | Recommendation | Reasoning |
|----------|---------------|-----------|
| forge / tool use | Either (nothink preferred) | think/nothink don't inject into forge |
| agent_recorded / agent loop | **think** | +3.9 pp improvement |
| code generation | **nothink** | think mode drops 80% → 20% |
| reasoning (ds4-eval, truthfulqa) | **think** | +11 pp, +4 pp respectively |
| knowledge (hellaswag, gsm8k) | **nothink** | hellaswag -17 pp with think |

**Default recommendation**: run nothink for coding tasks, think for reasoning/analysis.

## Server Bug Fixes Applied (image `dc20057e-cuda12`)

All three fixes are in the Docker image. Earlier images may be missing some:

### Fix 1: `call:verb{}` streaming detection (commit `658d016f`)
`sse_emitter.cpp` `find_tool_start()` only detected `<tool_call>` XML patterns.
Gemma4 emits `call:verb{}` plain-text format — Pattern B added to detect this.
Also includes C++17 compat fix for `starts_with` → `rfind`.
*Impact*: Gemma4 forge 0% → 20% (model behavior limits ceiling).

### Fix 2: forge `StepEnforcer` one-shot batch (vendored `step_enforcer.py`)
`StepEnforcer.check()` rejected one-shot batches (all tools in one response)
because required steps weren't "recorded" yet when the batch arrived. Fixed to
allow batches where required steps appear before the terminal tool.
*Impact*: Gemma4 basic_2step PASS (was failing despite correct tool ordering).

### Fix 3: Anthropic `tool_use` + `tool_result` context blocks (commit `dc20057e`)
`normalize_chat_messages()` in `http_server.cpp` silently dropped `tool_use`
(assistant tool call) and `tool_result` (user tool output) Anthropic-format
content blocks. Multi-turn tool conversations via `/v1/messages` always looped
(model never saw tool results → called same tool repeatedly).
*Impact*: **Qwen3.6 forge 0% → 100% (5/5)**, neutral for Gemma4.

## Decode Performance (throttled TDP, ~86-90 W)

| Model | Active params | Decode speed | Notes |
|-------|--------------|--------------|-------|
| Qwen3.6-27B (Q4_K_M) | 27B dense | ~24 tok/s | |
| Gemma4-26B-A4B (Q4_K_M) | 4B (sparse MoE) | ~66 tok/s | Most tokens are thinking |

At full "Best Performance" TDP, expect ~40-60% higher decode speeds.
Gemma4's speed advantage is mostly spent on thinking tokens; visible output
per turn is typically 50-200 chars.

## Autotune Sweep (2026-05-30)

Optimal Qwen3.6 config was determined by empirical sweep across:
- max_ctx: {65536, 98304} — 98304 wins on agent_recorded
- cache_type: {tq3_0, q8_0} — q8_0 @ 98K OOMs; tq3_0 required
- budget (ddtree): {16, 22, 32} — budget=16 wins for decode speed

See `qwen3.6-27b-sweep-runbook-bragi.md` and
`qwen3.6-27b-coding-agent-loop-sweep-bragi-2026-05-30.md` for sweep details.

## Known Gemma4 Limitations (not worth further tuning)

1. **Nothink ineffective**: Gemma4 thinks via `<|channel>thought` channel, not
   `<think>` tags. The `/no_think` prompt doesn't suppress it. Model burns full
   4096-token budget on hidden thinking even in nothink mode (12/26 agent_recorded
   cases have non-empty reasoning in nothink mode).
2. **Model refusals**: 2/26 agent_recorded cases return `given=refused`.
3. **code=0%**: `<|channel>thought` token leak in server (both think and nothink).
   Separate fix needed in `http_server.cpp` for channel token routing.
4. **forge multi-turn**: Gemma4 responds with text (not tool calls) when it receives
   tool results. Can only pass scenarios where all tools are emitted one-shot.
5. **reasoning_tokens=0, decode_ms=0**: Accounting/instrumentation bugs not ported
   from qwen35 backend.
