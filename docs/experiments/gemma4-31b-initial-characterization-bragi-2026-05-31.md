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
| `/v1/messages` tools | ✗ (model) | Server accepts requests, model ignores tool schema |
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

| area | n | pass_rate | wall_total | wall_median | Qwen3.6 ref | notes |
|------|---|-----------|------------|-------------|-------------|-------|
| forge | 30 | ~0% | skip | skip | 100% | Model generates prose instead of tool_use blocks (expected — consistent with gemma4-26b) |
| agent_recorded | 26 | **38.5%** | 629s | 23.4s | 38.5% | Re-run with `--max-tokens 512`; matches Qwen3.6 exactly |
| code | 10 | **70.0%** | 117s | 10.6s | 90% | Re-run with `--max-tokens 512`; -20pp vs Qwen3.6 |
| gsm8k | 100 | **95.0%** | 1922s | 11.8s | 81% | math, nothink — **+14pp vs Qwen3.6** |
| hellaswag | 100 | **79.0%** | 310s | 1.4s | 93% | MC knowledge — -14pp vs Qwen3.6 |
| truthfulqa-mc1 | 100 | **79.0%** | 438s | 0.4s | 82% | MC truthfulness — -3pp vs Qwen3.6 |
| longctx | 6 | **33.3%** (2/6) | 15s | — | 100% | frontier-2k,4k pass; 8k+ → HTTP 400 (template overflow at 32K) |
| smoke | 3 | 100% | 2s | — | 100% | confirmed passing |

## CRITICAL: DFlash Server Hang Bug

**Trigger**: Running a forge benchmark (Anthropic `/v1/messages`, `stream=false`) for Gemma4-31B
leaves a long conversation in-flight when the forge client is killed. When a subsequent
Anthropic-format request with 6 messages + 3 tool schemas arrives, the server enters an
**infinite GPU compute loop**: GPU shows 100% SM utilization but only 1% memory bandwidth
(compute-bound but not doing inference). The server's `/health` and `/v1/models` endpoints
remain responsive, but all inference requests queue indefinitely.

**What happened (2026-05-31)**:
- Forge-only test process killed → left `msg_0000000000000006` in-flight (253 tokens, 4096 max, 6 msgs, 3 tools, `stream=false`)
- Server stuck: 2h 27m of 100% SM / 1% mem bandwidth, no `DONE` ever logged
- All 26 agent_recorded cases and 10 code cases queued → all clients timed out at 300s with 0 tokens received
- Fix: `systemctl --user restart lucebox.service`

**Hypothesis**: DFlash speculative decoding has a bug in its rejection-sampling loop when
processing Gemma4-31B + Anthropic format + multi-message + tool schema context. The drafter
may propose tokens that cause an infinite verify/reject cycle in the CUDA kernel.

**Workaround**: Never run forge benchmarks (Anthropic `/v1/messages` with tool schema) while
other benchmarks may still be queued. Kill forge processes cleanly — force-kill leaves the
server request in-flight. If server seems stuck: restart service.

## Longctx Findings

Gemma4-31B at max_ctx=32768 hits HTTP 400 ("Bad Request" = prompt_tokens > max_ctx) for
frontier-8k and larger cases, even though the same max_ctx handles frontier-32k for Laguna.
The Gemma4 chat template has significant per-message overhead that causes the 8K approximation
to tokenize to >32768 real tokens. Frontier-2k (8.5s) and frontier-4k (6.1s) succeed.

**Effective practical prompt budget**: approximately 4K real tokens (the 4K case uses
`context_tokens_approx` well under the safety threshold; the 8K case overflows after template
expansion).

Compare to Qwen3.6-27B at max_ctx=98304: handles all frontier cases ≤ 32K easily.

## Performance Summary vs Qwen3.6-27B (bragi, nothink)

| area | Gemma4-31B | Qwen3.6-27B | delta |
|------|------------|-------------|-------|
| gsm8k | **95.0%** | 81% | **+14pp** ← standout strength |
| truthfulqa-mc1 | 79.0% | 82% | -3pp |
| hellaswag | 79.0% | 93% | -14pp |
| longctx | 33.3% | 100% | -67pp ← severe (context window limit) |
| agent_recorded | **38.5%** | 38.5% | 0pp — ties Qwen3.6 |
| code | **70.0%** | 90% | -20pp |

**Verdict**: Gemma4-31B has exceptional math (GSM8K +14pp) and matches Qwen3.6 on
agent_recorded. However it's significantly worse at coding (-20pp), MC knowledge (-14pp), and
long-context tasks (-67pp, hard-limited by the 32K context ceiling). For agent/coding workloads
on 24 GB VRAM, Qwen3.6-27B remains the preferred model. Gemma4-31B may be worth revisiting
for math-heavy or reasoning-intensive workloads once think mode is wired up.

## Key Operational Notes

- **Always run forge LAST or separately** for Gemma4-31B. Running forge before other benchmarks
  leaves Anthropic-format requests in-flight; if the forge client is killed, the server hangs
  indefinitely (100% SM / 1% mem GPU loop). Restart the service to recover.
- **Use `--max-tokens 512` for agent_recorded** (or any area where 4096 is overkill). At
  22 tok/s effective, 4096 tokens = 186s per case — close enough to the 300s timeout that
  slow DFlash acceptance rates can push individual cases over. 512 tokens → 23s per case.
- **longctx effective limit is ~4K tokens** at max_ctx=32768. The Gemma4 chat template
  expands prompts more aggressively than the 1.43× safety factor assumes; frontier-8k+ → HTTP 400.

## Next Steps

1. Try think mode (`<|think|>` reasoning) — `reasoning_supported=False` in props but model has the token
2. Compare vs Gemma4-26B-A4B on the same areas
3. Investigate DFlash server hang bug with speculative decoding team
4. Consider higher max_ctx (e.g. 65536) for non-MoE access patterns if VRAM allows
