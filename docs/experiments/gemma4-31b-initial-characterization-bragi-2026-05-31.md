# Gemma4-31B Initial Characterization — bragi — 2026-05-31

## Hardware

bragi: RTX 5090 Laptop MaxQ, 23 GB VRAM (24,463 MiB), WSL2, Windows Balanced (~86-90W TDP)

## Model

| field | value |
|-------|-------|
| preset | `gemma-4-31b` |
| target | `google_gemma-4-31B-it-Q4_K_M.gguf` — 20 GB |
| draft | `gemma-4-31B-it-DFlash-q8_0.gguf` — 1.6 GB |
| architecture | gemma4, 60 layers (dense 30.7B params) |
| context | 262,144 tokens native (256K) |
| embed | 5376 |
| reasoning | via `<|think|>` token |

## Server Config (initial sweep, 32K context, nothink)

```toml
[model]
preset = "gemma-4-31b"

[dflash]
budget = 8
max_ctx = 32768
cache_type_k = "tq3_0"
cache_type_v = "tq3_0"
think_max = 8192
prefix_cache_slots = 0
```

Image: `lucebox-hub:cuda12` (= `dc20057e-cuda12`)

## Server Capabilities

| capability | status | notes |
|-----------|--------|-------|
| DFlash speculative | ✓ active | `speculative_mode=dflash`, budget=8 |
| `/v1/messages` tools | ✓ working | forge area runs via Anthropic API |
| `/v1/chat/completions` tools | ✗ | `tools_supported=False` in props |
| Reasoning (`<|think|>`) | ? | `reasoning_supported=False` in props — arch wiring TBD |
| KV quantization | ✓ tq3_0 | Unlike Gemma4-26B-A4B which forces F16 |

## Differences vs Prior Models

| aspect | Gemma4-26B-A4B | Gemma4-31B | Qwen3.6-27B |
|--------|----------------|------------|-------------|
| params | 4B active / 26B total (MoE) | 30.7B dense | 27B dense |
| VRAM target | 17 GB | 20 GB | 17 GB |
| VRAM draft | 457 MB | 1.6 GB | 1.4 GB |
| Max safe ctx | 131K (F16 KV) | ~32K (tq3_0) | 98K (tq3_0) |
| KV quant | F16 forced (26B arch) | tq3_0 (31B arch) | tq3_0 |
| Reasoning | `<|think|>` | `<|think|>` | `<think>` |

## Benchmark Results (32K context, nothink)

Baseline: `bragi-rtx5090laptop-gemma4-31b-nothink-32k-2026-05-31`

*(Results to be filled in after run completes)*

| area | n | pass_rate | wall_total | wall_median | Qwen3.6 ref | notes |
|------|---|-----------|------------|-------------|-------------|-------|
| forge | 30 | TBD | TBD | TBD | 100% | via /v1/messages |
| agent_recorded | 26 | TBD | TBD | TBD | 38.5% | no system prompt |
| code | 10 | TBD | TBD | TBD | 90% | HumanEval-port |
| gsm8k | 100 | TBD | TBD | TBD | 81% | math, nothink |
| hellaswag | 100 | TBD | TBD | TBD | 93% | MC knowledge |
| truthfulqa-mc1 | 100 | TBD | TBD | TBD | 82% | MC truthfulness |
| longctx | 6 | TBD | TBD | TBD | 100% | 32K max |
| smoke | 3 | 100% | 2s | — | 100% | confirmed passing |

## Next Steps

1. Fill in benchmark results table above
2. Try think mode if reasoning API gets wired
3. Context window check: is 32K truly the max, or can we use more?
4. Compare vs Gemma4-26B-A4B on the same areas
