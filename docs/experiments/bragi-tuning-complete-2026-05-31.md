# Bragi auto-tuning: complete — 2026-05-31

All major tunables have been swept and validated on bragi (RTX 5090 Laptop
MaxQ, 23 GB VRAM, WSL2). This document is the final record.

## Optimal configuration

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
prefill_mode = "off"
lazy = false
```

**Image**: `dc20057e-cuda12` (lucebox-hub:cuda12, three server bug fixes applied)

## Tunable sweep summary

| tunable | values tested | winner | evidence |
|---------|--------------|--------|----------|
| budget (ddtree) | 16, 22, 32 | **16** | sweep 2026-05-30; 22/32 slower |
| max_ctx | 65536, 98304 | **98304** | +5pp agent_recorded; VRAM fits with tq3_0 |
| cache_type_k/v | q8_0, tq3_0 | **tq3_0** | q8_0 OOMs at 98K on 23 GB VRAM |
| fa_window | 0, 512, 1024 | **0** (full attn) | no quality gain with sliding window |
| prefix_cache_slots | 0, 32 | **0** | -19pp agent_recorded with 32 (bug in snapshot path) |
| pflash (prefill_mode) | off, auto | **off** | no-op with prefix_cache_slots=0; all chunks forced |
| think mode | nothink, think | **per-task** | see guidance table below |

## Think vs nothink guidance (Qwen3.6-27B)

| task type | recommendation | key deltas |
|-----------|---------------|------------|
| forge / tool use | nothink | –/– (think flag has no effect on forge runner) |
| agent / agent_recorded | **think** | +3.9pp on agent_recorded |
| reasoning (ds4-eval, truthfulqa) | **think** | +6.5pp ds4-eval, +4pp truthfulqa |
| code generation | **nothink** | think -60pp (format mismatch) |
| knowledge MC (hellaswag, gsm8k) | **nothink** | hellaswag -15pp, gsm8k -4pp with think |

## Benchmark results on dc20057e-cuda12 (nothink)

| area | pass_rate | wall_median |
|------|-----------|-------------|
| forge | **100%** (5/5) | 6.1s |
| agent | **100%** (4/4) | 11s |
| longctx | **100%** (6/6) | 40s |
| code | **90%** (9/10) | 1.5s |
| truthfulqa-mc1 | **80%** (80/100) | — |
| hellaswag | **88%** (88/100) | 0.5s |
| gsm8k | **86%** (86/100) | — |
| ds4-eval | **77.2%** (71/92) | 68.7s |
| agent_recorded | **42.3%** (11/26) | 26s |

Think mode results (from image 658d016f, unaffected by dc20057e fix):

| area | think pass_rate |
|------|----------------|
| ds4-eval | 81.5% |
| truthfulqa-mc1 | 84% |
| agent_recorded | 46.2% |
| hellaswag | 73% |
| gsm8k | 82% |
| code | ~20% |

## Server bug fixes applied in dc20057e-cuda12

Three fixes required for correct results on this image:

1. **`call:verb{}` streaming detection** (`658d016f`): Gemma4 forge 0% → 20%.
   `sse_emitter.cpp` only detected `<tool_call>` XML patterns; added Pattern B
   for Gemma4's `call:verb{}` plain-text format.

2. **`StepEnforcer` one-shot batch** (vendored `step_enforcer.py`): Allowed
   one-shot batches where required steps appear before terminal tool in a single
   response. Fixes Gemma4 basic_2step.

3. **`tool_use` + `tool_result` Anthropic blocks** (`dc20057e`): Qwen3.6 forge
   0% → 100%. `normalize_chat_messages()` silently dropped these block types,
   causing multi-turn tool conversations to loop infinitely.

## Known limitations (not tunable away)

- **prefix_cache_slots**: 32-slot server default causes -19pp regression on
  agent_recorded due to daemon snapshot path bug with tool-calling convos.
  Will stay at 0 until the underlying bug is fixed. See
  `qwen3.6-27b-prefix-cache-regression-bragi-2026-05-31.md`.

- **Gemma4 code=0%**: `<|channel>thought` token (id 100) leaks into output.
  Fix: `raw.starts_with("<|channel>")` in `http_server.cpp` lines 1534+1711.
  Requires image rebuild. Gemma4 is not the preferred model for this hardware.

- **Gemma4 nothink ineffective**: Model uses `<|channel>thought` channel,
  not `<think>` tags. `/no_think` prompt cannot suppress it.

- **pflash speed improvement**: Would require both prefix_cache_slots>0 AND
  the snapshot path bug to be fixed. Two-step dependency.

## Performance at throttled TDP (86-90W, Windows Balanced)

| model | decode speed | notes |
|-------|-------------|-------|
| Qwen3.6-27B Q4_K_M | ~24–25 tok/s | 27B dense, speculative budget=16 |
| Gemma4-26B-A4B Q4_K_M | ~66 tok/s | 4B active (sparse MoE); most are thinking |

At Windows "Best Performance" TDP, expect ~40-60% higher decode speeds.

## What was NOT swept (and why)

- **Sampling params** (temperature, top_k, top_p): Model-card defaults (temp=1.0,
  top_k=20, top_p=0.95) are the Qwen team's recommendations. Changing globally
  risks hurting some tasks to marginally improve others. Not swept.

- **think_max**: Currently 15488 tokens. Higher values could improve hard-reasoning
  quality but at proportional speed cost. At 24 tok/s, 15K tokens = 625s — already
  the practical ceiling for interactive use.

- **Gemma4 tuning**: Gemma4 is not the preferred model for this hardware (code=0%,
  agent_recorded 19.2% vs Qwen3.6 42.3%). Further tuning has low ROI until the
  `<|channel>thought` token-leak bug is fixed.
