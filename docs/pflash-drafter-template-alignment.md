# Drafter / target distribution alignment via closed-think prefill

## Problem

PR #274 (adaptive composition) shipped on `feat/pflash-drafter-ee7`, validating
13× prefill TPS and +47% decode TPS at long context. It surfaced a load-bearing
ceiling on the dflash decode side: spec-decode `accept_rate` was capped at
13–21% on the opencode harness and went to 0.0% on a peer-chat call. Composition
arm decode TPS (24.4 tok/s) therefore stayed below pflash-only (33.0 tok/s) —
the drafter overhead wasn't amortizing through acceptance.

## Diagnosis (the wrong hypothesis first)

The peer-chat conversation suggested "drafter conditioned on a different chat
template than the target." Three Phase-1 Explore agents traced the code and
showed that framing is architecturally wrong:

- Both target and drafter receive the **same** `effective_prompt` token IDs at
  prefill. The chat template is applied **once** on the target side at
  `server/src/server/http_server.cpp:996-1014`, tokenized with the target's
  tokenizer at `:1014`, then flows to both target and drafter via
  `gen_req.prompt = effective_prompt` at `:1265`.
- The drafter `dflash-draft-3.6-q4_k_m.gguf` does **not** apply any chat
  template at runtime. `server/src/draft/draft_gguf_loader.cpp` doesn't read
  the `tokenizer.chat_template` GGUF metadata key.

A `--draft-chat-template` flag would fix nothing — there is no drafter-side
template-application code path to redirect.

## Diagnosis (the actual root cause)

The drafter GGUF **does** ship the official Qwen3.6 chat template as
`tokenizer.chat_template` metadata. That template appends
`<think>\n\n</think>\n\n` after `<|im_start|>assistant\n` when
`enable_thinking=false`. The drafter was distilled with that closed-think
suffix in its training distribution — every assistant turn it predicts
expects that prefix.

The target's Unsloth Qwen3-Coder template (`project_unsloth_jinja_template_solves_tool_call`
in memory) does **not** append that suffix. So at the moment spec-decode
predicts the next token after `<|im_start|>assistant\n`:

- drafter's distribution expects `<think>` literal tokens
- target's distribution expects the actual answer

Drafter proposes `<think>...`, target rejects, falls back to AR. Repeat at
every position. `accept_rate` ≈ 0%.

## Fix

Make the **target's render** match the drafter's training distribution.
`render_chat_template_jinja` now appends `<think>\n\n</think>\n\n` after a
bare `<|im_start|>assistant` marker when **all three** of these hold:

1. `arch_hint == ChatFormat::QWEN3` (gated to Qwen3-family — qwen35, qwen35moe;
   Laguna / Gemma4 don't use ChatML tokens and must not be touched)
2. `!enable_thinking`
3. The rendered prompt ends with the bare assistant marker (tolerant of
   trailing whitespace variants: `\n`, `\n\n`, trailing space)

Condition (3) prevents double-appending when a user-supplied template already
emits the closed-think suffix.

## Multi-arch safety

`chat_format_for_arch()` in `server/src/server/chat_template.cpp` returns:
- `ChatFormat::QWEN3` for `qwen3`, `qwen35`, `qwen35moe`
- `ChatFormat::LAGUNA` for `laguna`
- `ChatFormat::GEMMA4` for `gemma4`

The suffix only fires for `QWEN3`. A new test
(`test_chat_format_for_arch_qwen35moe_returns_qwen3`) locks the qwen35moe →
QWEN3 inheritance so a future arch-enum addition doesn't silently flip
behavior. Tests also lock the Laguna/Gemma4 no-append case and the
no-double-append guard.

## Expected impact

- `accept_rate` lifts from 13–21% (and 0% on peer-chat) on Qwen3.6 dense with
  Unsloth Qwen3-Coder template. Threshold for declaring the fix worked:
  non-zero peer-chat accept_rate AND opencode harness accept_rate ≥30% on at
  least 2 of 3 turns from Round 5b D.
- Composition arm decode TPS rises above pflash-only on long-generation
  workloads (currently 24.4 vs 33.0; the gap exists because spec-decode
  amortization is bounded by accept_rate).
- davide221's qwen35moe `chat CACHE` hang (issue #280) likely has the same
  root cause via the same code path — qwen35moe inherits ChatFormat::QWEN3
  and the suffix will fire there too.

## Out of scope

The sibling commits on `fix/qwen36-claude-code-tool-calling` (target-side
tool-format normalization, scrub/truncate, Anthropic→Qwen tool shape,
param-name aliasing) ship as PR #276. They are not drafter alignment — they
are independent target-side tool-formatting improvements.
