# bragi (RTX 5090 Laptop) — Qwen3.6-27B — Level 2 force-close partial bench

**Date**: 2026-05-23
**Image**: `lucebox-hub:cuda12 = cea5704f6b26` (built from `a1144dd`)
**Server**: native dflash_server (Python server.py deleted in `43ad46b`)
**Branch**: `integration/props-uv-squared-clean`

## Configuration

- `--think-max-tokens` = 10000 (Level 1 phase-1 cap)
- `--default-max-tokens` = 16000 (request-omitted default)
- `--hard-limit-reply-budget` = 512 (Level 2 force-close threshold)
- `</think>` token resolved at startup → ID **248069** (Qwen3.6 single special token)
- Spec-decode with tail-off to AR when budget tightens (`a1144dd`)
- `DFLASH_THINK_MAX` = 15488 via entrypoint env (sindri's `9017c92`)
- `DFLASH_BUDGET=22, DFLASH_MAX_CTX=98304, DFLASH_LAZY=1`

## Partial results (6/92 cases, bench in flight)

| # | source | id | given | correct | PASS | wall |
|---|---|---|---|---|---|---|
| 1 | GPQA Diamond | recNu3MXkvWUzHZr9 | **B** | B | ✅ | 263.33s |
| 2 | SuperGPQA | 001b51d76b… | C | C | ✅ | 35.74s |
| 3 | AIME2025 | aime2025-01 | 70 | 70 | ✅ | 205.82s |
| 4 | GPQA Diamond | recoiTJPGUmzAkief | C | C | ✅ | 378.43s |
| 5 | SuperGPQA | b7e20eac… | J | J | ✅ | 273.75s |
| 6 | AIME2025 | aime2025-16 | 468 | 468 | ✅ | 229.44s |

**6/6 PASS, format=True hint=True on every row.** `[budget-hook]` fired **10 times** across these 6 cases (some cases trigger force-close in two phases).

## Flip vs Level 1 — case 1 specifically

`recNu3MXkvWUzHZr9` (GPQA Diamond, LMC astronaut at 0.99999987c, correct answer B):

| Path | Result | Cause |
|---|---|---|
| Native vidar (in-process ds4_eval) | PASS B | in-process force-close at hard_limit |
| OpenRouter Qwen3.6-27B | PASS B | full FP16 inference, naturally writes answer |
| Local v1 lucebox (Python, broken 4096 cap) | FAIL ? (format_error) | bench's `max_tokens=4096` truncated mid-reasoning |
| C++ L1 (phase-1/phase-2 reprompt) | FAIL D | phase-2's "Final answer: " prefill landed model on a shallow guess |
| C++ L2 (force-close, this snapshot) | **PASS B** | model gets remaining 512 tokens to write answer with KV intact after injected `</think>` |

## Comparison context

| Bench | Pass rate (92 cases) | Notes |
|---|---|---|
| ds4-eval native (m2 ultra, in-process force-close) | **76/92 = 82.6%** | gold standard |
| vidar HTTP via lucebox bench v2 (merged) | 75/92 = 81.5% | matches native within 1 case |
| OpenRouter Qwen3.6-27B (lucebox bench v2) | 58/92 = 63.0% | no force-close at backend |
| C++ L2 (this run, in progress) | **6/6 so far, full bench pending** | aiming for ≥native parity |

## Commits in this snapshot

Top of `integration/props-uv-squared-clean`:

```
a1144dd perf(cpp-server,L2): spec-decode tail-off to AR instead of full bypass
3c8a218 Merge origin/main into integration/props-uv-squared-clean
c7c4bcb feat(cpp-server,bench): emit reasoning aliases per multi-dialect spec
43ad46b chore(dflash): drop Python server.py and tests that import from it
5059fea feat(cpp-server): Level 2 in-process force-close on AR path
aacb9bb docs(thinking-budget): document multi-dialect response aliasing
a44a7bc fix(cpp-server): forward Qwen3.6 <think>/</think> tokens to emitter
53c85c5 fix(entrypoint): priority-order draft resolution to avoid safetensors
3e8323d fix(cpp-server): phase-2 gate diagnostics + entrypoint draft-path + max_tokens banner
2a89e62 fix(cpp-server): pre-open <think> when enable_thinking=true (Qwen3.6)
```

## What's still open (C++ server)

1. Strip `[phase2-gate]` diagnostic logging
2. Fix `finish_reason="length"` on phase-1 cap hit
3. Apply stop sequences to phase-2 content
4. Port Level 2 to laguna + gemma4 backends
5. Multi-token `</think>` close-tag support (disables L2 with warning currently)
6. Soft-close mode (voluntary close when `</think>` in top-K)
7. Streaming phase-2 reprompt

This snapshot will be replaced with the canonical `lucebox profile --export-snapshot` artifact once the full 92-case bench completes (~4-6h from launch).
