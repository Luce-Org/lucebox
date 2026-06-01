# Laguna-XS.2 Initial Characterization — bragi — 2026-05-31

## Hardware

bragi: RTX 5090 Laptop MaxQ, 23 GB VRAM (24,463 MiB), WSL2, Windows Balanced (86-90W TDP)

## Model

| field | value |
|-------|-------|
| preset | `laguna-xs.2` |
| target | `laguna-xs2-Q4_K_M.gguf` — 20.3 GB |
| speculator | `draft/laguna-xs2-speculator/model.safetensors` — 1.2 GB |
| speculator source | `poolside/Laguna-XS.2-speculator.dflash` |
| architecture | 40 layers, embed=2048, 8 GQA KV heads, head_dim=128 |
| context | 131,072 tokens native |
| MoE | 3B active / 33B total |

## Server Config (initial sweep, 32K context)

```toml
[model]
preset = "laguna-xs.2"
draft_file = "laguna-xs2-speculator"

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
| Reasoning (`<think>`) | ✗ (arch gap) | `reasoning_supported=False` — server not wired |
| Sampling defaults | temp=0.6, top_k=50 | code-model conservative |

## VRAM Budget

| item | MiB |
|------|-----|
| Model (target GGUF) | ~20,274 |
| Speculator (safetensors) | ~1,229 |
| KV cache (tq3_0, 32K) | ~960 |
| **Total used** | **22,955** |
| GPU total | 24,463 |
| **Free** | **1,183** |

KV formula: `40 layers × 8 KV heads × 128 dim × 2 (K+V) × 3/8 bytes × context_len`
= 30,720 bytes/token ≈ 30 KB/token at tq3_0.

### Context window feasibility

| max_ctx | KV (MiB) | Free after | feasible? |
|---------|----------|------------|-----------|
| 32,768 | 960 | 1,183 | ✓ working |
| 40,960 | 1,200 | 1,063 | ✓ expected |
| 49,152 | 1,440 | 823 | likely ok |
| 57,344 | 1,680 | 583 | tight |
| 65,536 | 1,920 | 343 | risky |

## Performance

Decode speed at temp=0.6 (forge cases, 40-60 token outputs):
- **~60-63 tok/s effective** (with DFlash speculator, budget=8)
- Qwen3.6-27B reference: ~25 tok/s → Laguna is **2.4× faster** on short outputs
- Speculator benchmark (temp=0): 125 tok/s vs 78 tok/s without (+60%)

## Benchmark Results (32K context, nothink)

Baseline: `bragi-rtx5090laptop-laguna-xs2-speculator-nothink-32k-2026-05-31`

| area | n | pass_rate | wall_total | wall_median | Qwen3.6 ref | notes |
|------|---|-----------|------------|-------------|-------------|-------|
| forge | 30 | 0.0% | 654s | 13.5s | 100% | Model doesn't emit tool_use blocks — fundamental limit |
| agent_recorded | 26 | 50.0% | 324s | 2.4s | 38.5% | +11.5pp vs Qwen3.6 |
| code | 10 | 20.0% | 12s | 1.0s | 90% | FIM format mismatch, not capability gap |
| gsm8k | 100 | 93.0% | 330s | 3.0s | 81% | +12pp vs Qwen3.6, 5× faster |
| hellaswag | 100 | 80.0% | 47s | 0.4s | 93% | -13pp vs Qwen3.6 |
| truthfulqa-mc1 | 100 | 69.0% | 61s | 0.4s | 82% | -13pp vs Qwen3.6 |
| longctx | 6 | 83.3% (5/6) | 52s | 4.4s | 100% | frontier-64k FAILS (65K > 32K ctx) |
| smoke | — | skip | — | — | 100% | tools unsupported in chat completions |

### Notes on outliers

- **forge 0%**: Laguna outputs plain text instead of structured `tool_use` blocks even when called via `/v1/messages`. Poolside's code model is FIM/completion-specialized, not instruction-tuned for Anthropic tool format.
- **code 20%**: Laguna uses FIM completion style — outputs only the function body starting mid-function, not a complete parseable Python file. luce-bench reports `given=parse_fail`. This is a prompt format mismatch; actual coding quality is likely much higher.
- **gsm8k 93%**: Strong math. Fastest area by far (3.0s median vs Qwen3.6's ~17s).
- **longctx frontier-64k**: 65,212-token input exceeds max_ctx=32768. The 5th pass was `frontier-32k` (32,449 tokens input, 43-token output → 32,492 total < 32,768).

## Differences vs Qwen3.6-27B

| aspect | Qwen3.6-27B | Laguna-XS.2 |
|--------|------------|-------------|
| VRAM | 17.9 GB target + 1.4 GB draft | 20.3 GB target + 1.2 GB spec |
| Decode speed | ~25 tok/s | ~60 tok/s (2.4× faster) |
| Max safe ctx | 98,304 (96K, tq3_0) | ~49K-57K (tq3_0) |
| Tool support | ✓ full | ✗ (chat completions) / ✓ (/v1/messages) |
| Reasoning | ✓ wired (think_max=15488) | ✗ not wired |
| Specialization | general | code-optimized |
| MoE | no (dense 27B) | yes (3B active) |

## Next Steps

1. Fill in benchmark results table above
2. Context window sweep: try max_ctx=40960, 49152, 57344
3. Budget sweep: compare budget=4, 8, 16 on Laguna
4. Evaluate think mode (once server reasoning support is wired)
5. Characterize Gemma4-31B and Qwen3.6-MoE on bragi
