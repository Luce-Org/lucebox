# Thinking budget — separate think vs reply token caps

A design spec for `dflash_server`'s handling of "thinking" requests:
prompts where the model is expected to produce an internal reasoning
trace before its visible reply. The spec covers the request opt-in
(including per-request budget controls), the configuration surface,
the two close strategies (Level 1 and Level 2), the multi-dialect
response shape, and the close-kind taxonomy.

## 1. Background

A reasoning-capable model wraps its internal scratch work in a
delimited block — by convention `<think> … </think>` for Qwen-family
chat templates, and equivalent tags for other architectures. The
text inside is the **reasoning trace**; the text after `</think>` is
the **visible reply**.

A single combined token cap (`max_tokens` on the wire) is not enough
to control these requests:

- On hard reasoning prompts the model can spend its entire budget
  inside the `<think>` block and never emit `</think>`. The response
  arrives with no parseable answer.
- Even when the model does close `</think>` on its own, a tight cap
  can leave it with no remaining tokens to write the actual answer.

We need two independent caps — one on reasoning length and one on
the combined output — plus a server-side mechanism that *forces*
the model out of `<think>` if the reasoning cap is reached without
the model self-closing. That contract is the **thinking budget**.

## 2. Terminology

- **Phase 1 — reasoning.** Generation between the opening `<think>`
  and the model's `</think>`. Output is reasoning text.
- **Phase 2 — content.** Generation after `</think>`. Output is the
  visible reply.
- **Budget envelope.** The set of caps a thinking-enabled request
  agrees to be governed by: phase-1 cap, combined cap, and reply-
  budget reserve. See §3.
- **Close kind.** How `</think>` ended up in the stream. See §6.

## 3. Configuration

Server-side configuration establishes the **ceilings** that bound
every request's budget envelope. Per-request fields (see §4) may
request *tighter* values than the ceilings, but never looser — this
gives operators an unconditional resource-protection guarantee while
letting clients tune for their use case (short chat vs. deep
reasoning).

### 3.1 Configuration sources

The server resolves each knob from the first source that provides a
value, in this order:

1. **Explicit CLI flag** (e.g. `--think-max-tokens N`).
2. **Model card sidecar.** A JSON file at
   `share/model_cards/<name>.json`, where `<name>` is the loaded
   GGUF's `general.name` metadata normalized to lowercase with
   spaces replaced by `-`. Carries values from the upstream model
   card (HuggingFace README or `generation_config.json`).
3. **Per-family fallback table**, built into the C++ server, keyed
   on the detected architecture (e.g. `qwen35`, `gemma4`, `laguna`).
   A coarse safety net for known families when no sidecar is shipped.
4. **Hard fallback**:
   `default_max_tokens=16000`, `hard_limit_reply_budget=4096`
   (raised from 512 on 2026-05-25 — see `hard_limit_reply_budget` row in
   the sidecar field table below for why),
   `think_max_tokens = default_max_tokens − hard_limit_reply_budget`.

The resolution is reported in the startup banner so operators can
see which source supplied each value.

### 3.2 Server CLI

```
dflash_server \
  --think-max-tokens 32256 \         # Phase-1 ceiling
  --default-max-tokens 32768 \       # Combined ceiling when the
                                     # request omits max_tokens
  --hard-limit-reply-budget 4096 \   # Reply-reserve ceiling (default)
  --reasoning-effort-low 4032 \      # Phase-1 budget for effort=low
  --reasoning-effort-medium 16128 \  # Phase-1 budget for effort=medium
  --reasoning-effort-high 32256 \    # Phase-1 budget for effort=high
  --reasoning-effort-x-high 56832 \  # Phase-1 budget for effort=x-high
  --reasoning-effort-max 81408 \     # Phase-1 budget for effort=max
                                     # (each capped at --max-ctx − --hard-limit-reply-budget)
```

CLI flags always win. Omit any flag to take the value from the
model card sidecar, family fallback, or hard fallback in turn.

### 3.3 Model card sidecar

Each known model has a sidecar JSON at
`share/model_cards/<name>.json`. The file carries values transcribed
from the upstream model card so future-us can reason about
provenance:

```json
{
  "name": "Qwen3.6 27B",
  "source": "https://huggingface.co/Qwen/Qwen3.6-27B",
  "verified_at": "2026-05-23",
  "max_tokens": 32768,
  "complex_problem_max_tokens": 81920,
  "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
  "reasoning_effort_tiers": {
    "low":    4032,
    "medium": 16128,
    "high":   32256,
    "x-high": 56832,
    "max":    81408
  }
}
```

Fields:

| Field | Meaning |
|---|---|
| `name` | Display name. Informational; the filename is what matters for lookup. |
| `source` | URL of the upstream card we transcribed. |
| `verified_at` | ISO date the values were last checked against the source. |
| `max_tokens` | The card's standard recommended combined cap. Drives `default_max_tokens`. |
| `complex_problem_max_tokens` | Optional. The card's recommendation for hard reasoning / benchmark workloads. Drives the `x-high` and `max` effort tiers, which sit *above* `default_max_tokens` when this field is present — they are admissible as long as they fit under `max_ctx − hard_limit_reply_budget`. If omitted, both collapse to the `high` tier value. |
| `hard_limit_reply_budget` | Optional. Tokens reserved post-`</think>` for the visible answer phase, used both to derive `think_max_tokens = max_tokens − hard_limit_reply_budget` and as the force-close trigger inside `do_ar_decode` / `do_spec_decode` (when `n_gen − generated ≤ hard_limit_reply_budget`, the engine overrides the next sampled token with `</think>`). Default 4096 (raised from 512 on 2026-05-25). The original 512 came from `ds4_eval.c`, sized for DeepSeek-V4-flash's terse style, but it silently truncated almost every other model mid-answer — bench results from `server/docs/experiments/gemma4-26b-thinking-control-2026-05-25.md` showed every force-closed thinking probe getting cut off mid-coordinate-geometry-proof at 512. Without priors on a specific model, 4096 is the safer default; terse models should override down. Qwen3.6, Gemma 4 26B, Gemma 4 31B all ship 4096 in their sidecars. |
| `sampling` | Recommended sampler params. Used as defaults when the request doesn't pin sampler values. |
| `reasoning_effort_tiers` | Explicit phase-1 budgets per tier. Override any computed default. Whichever tiers are present win; missing tiers fall through to the computed defaults below. |

If the sidecar omits `reasoning_effort_tiers`, tier values are
computed from `max_tokens` and `complex_problem_max_tokens`:

| Tier | Default formula |
|---|---|
| `low` | `think_max × 0.125` |
| `medium` | `think_max × 0.5` |
| `high` | `think_max × 1.0` (= ceiling derived from `max_tokens`) |
| `x-high` | `(think_max + complex_think_max) / 2` |
| `max` | `complex_think_max` |

Where `think_max = max_tokens − hard_limit_reply_budget` and
`complex_think_max = complex_problem_max_tokens − hard_limit_reply_budget`
(or `think_max` if the card has no complex recommendation).

The explicit `reasoning_effort_tiers` field exists because the
ratio-based defaults don't fit every model. A smaller model that
caps at 8192 tokens has a very different tier curve from Qwen3.6's
32768/81920 envelope, and the model card author is in a better
position to pick sensible numbers than a global formula.

For the Qwen3.6 example above (`max_tokens=32768`,
`complex_problem_max_tokens=81920`), the resolved tiers are
`low=4032, medium=16128, high=32256, x-high=56832, max=81408`.
The `x-high` and `max` values exceed `default_max_tokens`, but they
are *phase-1 budgets* — clients that want to use them in full must
also pass an explicit `max_tokens` ≥ `tier_value +
hard_limit_reply_budget` on the request. With smaller `max_tokens`,
the request parser narrows the effective phase-1 cap to fit (see
§4.4). The tiers stay distinct rather than collapsing to `high`
because the ceiling that bounds them is `max_ctx`, not
`default_max_tokens`.

### 3.4 Hard fallback

When no sidecar matches the loaded model and no family fallback
applies, the server uses the `antirez/ds4 ds4_eval.c` reference
values:

| Knob | Hard fallback | Role |
|---|---|---|
| `--default-max-tokens` | 16000 | Combined ceiling |
| `--hard-limit-reply-budget` | 512 | Reply-reserve ceiling |
| `--think-max-tokens` | 15488 | Phase-1 ceiling (= 16000 − 512) |

Effort tiers in this configuration: `low=1936`, `medium=7744`,
`high=15488`, `x-high=15488`, `max=15488` (the last two collapse to
`high` because no `complex_problem_max_tokens` is defined).

### 3.5 Effort-tier invariants

The server enforces these invariants at startup and clamps with a
warning if violated:

- `low ≤ medium ≤ high ≤ x-high ≤ max`
- `max ≤ max_ctx − hard_limit_reply_budget`

The server's `--max-ctx` is the absolute ceiling for any single
request — including its phase-1 portion. Effort tiers are *phase-1
budgets*, not combined budgets; a tier value larger than
`default_max_tokens` is well-defined. It just means a client that
wants to use that tier's full budget needs to pass an explicit
`max_tokens` ≥ `tier_value + hard_limit_reply_budget` on the
request. With smaller `max_tokens`, the request parser narrows the
effective phase-1 cap to `min(tier_value, request.max_tokens −
hard_limit_reply_budget)` (see §4.4).

A request that asks for an effort tier exceeding the model's
ceiling (e.g. `effort: "max"` on a model whose card has no
`complex_problem_max_tokens`) gets the `high` value with no error.

## 4. Request shape

There are two equivalent ways a client opts into the budget envelope.
Both unlock Level 1, Level 2, and `finish_details` emission.

### 4.1 Anthropic-style `thinking`

```json
{
  "model": "...",
  "messages": [...],
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 4000,
    "reply_budget":  300
  }
}
```

| Field | Meaning |
|---|---|
| `thinking.type` | `"enabled"` activates the envelope; anything else (or omitting `thinking`) keeps the legacy single-cap behaviour. |
| `thinking.budget_tokens` | Optional. Client-preferred phase-1 cap. Effective value = `min(budget_tokens, --think-max-tokens)`. Omit to use the server default. |
| `thinking.reply_budget` | Optional. Client-preferred reply reserve for Level 2 force-close. Effective value = `min(reply_budget, --hard-limit-reply-budget)`. Omit to use the server default. |

### 4.2 OpenAI Responses-style `reasoning.effort`

```json
{
  "model": "...",
  "input": "...",
  "reasoning": {"effort": "medium"}
}
```

| Field | Meaning |
|---|---|
| `reasoning.effort` | One of `"low"`, `"medium"`, `"high"`, `"x-high"`, or `"max"`. Each value selects a server-configured phase-1 budget (see §3) and activates the envelope. |

`reasoning.effort` is the simpler shape for clients that don't want
to pick a token number. The effective phase-1 budget is the
`--reasoning-effort-<tier>` value at the chosen tier; the reply
reserve falls back to `--hard-limit-reply-budget`.

The five-tier vocabulary is a dflash extension to the
OpenAI Responses three-tier (`low | medium | high`) standard.
Clients that send only OpenAI-standard values continue to work; the
extra tiers (`x-high`, `max`) let clients opt in to the model card's
complex-problem budget when the prompt warrants it.

An unknown tier value falls back to `high` rather than erroring, so
clients that send a future tier (e.g. `"ultra"`) get sensible
behaviour instead of a 400.

### 4.3 Combining the two

If a request sets **both** `thinking.budget_tokens` and
`reasoning.effort`, `thinking.budget_tokens` wins (it is the more
specific control). The effort tier still selects defaults for any
unspecified `thinking.*` fields. This keeps mixed-dialect clients
predictable and lets per-request fine-tuning sit on top of a coarse
effort knob.

### 4.4 Clamping rules

All per-request budget fields clamp to the server ceiling — clients
can ask for *less* than the operator-configured ceiling but never
*more*:

| Per-request field | Clamp |
|---|---|
| `thinking.budget_tokens` | `min(requested, --think-max-tokens)` |
| `thinking.reply_budget` | `min(requested, --hard-limit-reply-budget)` |
| `max_tokens` (combined cap) | `min(requested, --default-max-tokens)` |

The server emits a single per-request log line whenever a clamp
fires, recording requested-vs-effective values for both fields. No
error response — clamping is silent at the wire to preserve OpenAI/
Anthropic protocol compatibility.

When `reasoning.effort` is set, the request's effective phase-1
cap is `min(effort_tier_value, request.max_tokens −
hard_limit_reply_budget)`. The effort tier value is the server
configuration looked up from the resolved model card (or CLI
override); the per-request narrowing accommodates clients that
choose a tier (e.g. `"max"`) without also passing an explicit
`max_tokens`. If the client wants to use the full tier budget,
they must also pass a large enough `max_tokens` — otherwise the
effective cap silently narrows to what fits in `max_tokens` after
reserving the reply budget. This narrowing is logged once at info
level (not a warning — it is normal and expected behaviour).

### 4.5 Why client-side controls are bounded, not full overrides

A previous design allowed clients to override the server budget
entirely. The footgun was: middleboxes that did not understand the
new fields silently dropped them, leaving requests to hit the
server's combined `max_tokens` as their only cap — invariably
truncating mid-reasoning and producing artificially low quality
numbers in cross-server benchmarks.

Clamping to the server ceiling resolves this asymmetrically: if a
middlebox drops the per-request field, the server falls back to its
configured default (which is a reasonable production policy), not to
the much-larger combined cap. Clients still get useful behaviour;
nobody silently truncates mid-thought.

## 5. Close strategies

When a request opts into the budget envelope the server uses one of
two strategies to ensure the response contains a visible reply, in
order of preference. Both are independent of the model architecture
in their contract; their implementation differs per backend.

### 5.1 Level 1 — phase-2 reprompt

When the daemon finishes phase-1 generation and `</think>` did not
appear in the stream, the server constructs a fresh prompt:

```
<original prompt tokens>
<phase-1 reasoning tokens>
</think>

Final answer:
```

It then runs a second daemon call against that prompt for at most
`max_tokens − phase1_emitted` more tokens and appends the result as
the visible reply.

Level 1 works on any backend; it does not require sampling-loop
integration. Its cost is one extra prefill of the phase-1 reasoning,
which dominates for long traces.

### 5.2 Level 2 — in-process force-close

When supported by the backend (currently Qwen3.5/3.6, Gemma4, Laguna),
the server avoids the phase-2 reprompt by overriding sampling in the
generation loop:

- Track the number of tokens generated since entry to the AR loop.
- When `(n_gen − generated) ≤ --hard-limit-reply-budget`, the
  remaining headroom is dedicated to the visible reply. Override the
  next sampled token with the tokenizer's `</think>` close-tag.
- Close tags that tokenize to multiple ids (e.g. DeepSeek/Laguna,
  where `</think>` is `[1718, 37947, 32]`) are injected as a multi-
  token sequence: each subsequent loop iteration overrides one more
  token until the sequence is complete. Single-token close tags
  (Qwen3.6 `</think>` = id 248069) finish in one override.
- After the close sequence, normal sampling resumes. The model
  continues from a still-hot KV cache and writes the visible reply
  naturally, with `--hard-limit-reply-budget` tokens of headroom.

Level 2 is strictly cheaper than Level 1 (no reprompt, no second
prefill, KV cache preserved) and produces a higher-quality reply
because the model's reasoning context is still in-frame when it
writes the answer.

When a Level 2-capable backend serves a thinking-enabled request,
Level 2 fires first. Level 1 remains as a fallback for backends that
do not yet implement the BudgetHook, and for safety in case Level 2
encounters an unexpected state.

### 5.3 Budget arithmetic

In Level 2 the budget check runs against tokens **generated in the
current AR loop**, not against the absolute KV position:

```
generated = committed_now − committed_at_entry
remaining = n_gen − generated
if remaining ≤ effective_reply_budget: force-close
```

Where `effective_reply_budget` is the per-request `thinking.reply_budget`
clamped to `--hard-limit-reply-budget` (see §4.4), and `n_gen` is the
effective phase-1 cap: `thinking.budget_tokens` clamped to
`--think-max-tokens` if set, otherwise the `reasoning.effort` tier value
narrowed by `request.max_tokens − hard_limit_reply_budget` (see §4.4).

The generated-since-entry frame matters because `committed_now`
includes the prompt length and any tokens already committed before
AR took over (e.g. when the spec-decode path tails off into AR for
the final stretch). Without the offset the check would fire
`prompt_len` tokens early and could go negative after spec-decode
tail-off, force-closing immediately as AR began.

## 6. Response shape

### 6.1 Reasoning text — multi-dialect aliases

Different reasoning-capable APIs put the reasoning trace under
different keys. There is no agreed-upon standard; each provider
picked one shape and tooling has fragmented around it.

| API | Reasoning text field | Reasoning-token count field |
|---|---|---|
| OpenAI o1/o3 | not exposed (tokens are hidden) | `usage.completion_tokens_details.reasoning_tokens` |
| Anthropic Claude | `content[]: {type:"thinking", thinking:"...", signature:"..."}` (typed block) | `usage.thinking_tokens` |
| DeepSeek R1 | `message.reasoning_content` (flat string) | inferred from totals |
| Qwen3 native | inline `<think>...</think>` in `message.content` | not exposed |
| OpenRouter | `message.reasoning` (flat) + `message.reasoning_details[]` (typed-block list) | `usage.completion_tokens_details.reasoning_tokens` |

dflash_server emits the reasoning text under **all** of the flat-
string names plus the typed-block list, and the OpenAI-shaped token
count, so any client written against any of these shapes works
without per-server remapping:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Final visible answer.",
      "reasoning_content": "Phase-1 reasoning text…",
      "reasoning":          "Phase-1 reasoning text…",
      "reasoning_details": [
        {"type": "reasoning.text", "text": "Phase-1 reasoning text…"}
      ]
    },
    "finish_reason": "stop",
    "finish_details": {
      "close_kind": "natural",
      "thinking_tokens": 8421,
      "content_tokens":  312,
      "total_tokens":    8733
    }
  }],
  "usage": {
    "prompt_tokens":     201,
    "completion_tokens": 8733,
    "total_tokens":      8934,
    "completion_tokens_details": {
      "reasoning_tokens": 8421
    }
  }
}
```

Field semantics:

- **`message.content`** — the visible reply (post-`</think>` text).
  Standard OpenAI Chat Completions field.
- **`message.reasoning_content`** — flat string with the full
  reasoning text. DeepSeek R1 convention. Primary field; tooling
  that knows only one of these field names should know this one.
- **`message.reasoning`** — same string as `reasoning_content`,
  under OpenRouter's normalized name.
- **`message.reasoning_details`** — a list of typed reasoning
  blocks. Today always exactly one `{type:"reasoning.text", text:…}`
  block carrying the full reasoning. The list shape leaves room to
  add phase-1/phase-2 splits, Anthropic-style signature fields, or
  per-stage metadata in a future version without breaking clients.
- **`usage.completion_tokens_details.reasoning_tokens`** — count of
  tokens attributed to reasoning. Matches OpenAI o1/o3's location
  and OpenRouter's normalization.
- **`finish_details`** — see §6.2.

The three `message.*` reasoning fields carry identical strings. They
are emitted together; clients should not assume they will diverge.

### 6.2 `finish_details`

When a request opts into the budget envelope, the response carries
an additional `finish_details` object alongside the standard OpenAI
`finish_reason`:

```json
"finish_details": {
  "close_kind":      "natural" | "hard",
  "thinking_tokens": <int>,
  "content_tokens":  <int>,
  "total_tokens":    <int>
}
```

- `close_kind` — see §7.
- `thinking_tokens` — tokens generated while the model was inside
  the `<think>` block. Equal to `usage.completion_tokens_details.reasoning_tokens`.
- `content_tokens` — tokens generated for the visible reply, summed
  across phase-1 (post-`</think>` if the model self-closed early)
  and phase-2 (Level 1 reprompt output).
- `total_tokens` — `thinking_tokens + content_tokens`.

`finish_reason` continues to follow OpenAI semantics
(`stop` / `length` / `tool_calls`). `finish_details` is additive:
clients that don't know about it ignore it.

`finish_details` is omitted when the request did not opt into the
budget envelope (no `thinking:{type:"enabled"}`).

### 6.3 Response timings

When the server completes a request, the response's `usage` block
carries a `timings` object with per-request performance metrics:

```json
"usage": {
  "prompt_tokens": 256,
  "completion_tokens": 1024,
  "total_tokens": 1280,
  "completion_tokens_details": { "reasoning_tokens": 512 },
  "timings": {
    "prefill_ms": 234.5,
    "decode_ms": 2456.7,
    "decode_tokens_per_sec": 41.6
  }
}
```

- `prefill_ms` — wall time spent processing the input prompt before
  generating the first output token (KV cache fill). Excludes queue
  and scheduling overhead.
- `decode_ms` — wall time spent generating output tokens
  (`completion_tokens` of them). Includes speculative-decode overhead.
- `decode_tokens_per_sec` — `completion_tokens / (decode_ms * 0.001)`.
  The model's effective throughput on this request. Emitted as
  `0.0` when `decode_ms` is zero (prefill-only / count-tokens paths)
  rather than `null` / `NaN` so JSON parsers don't trip.

These fields are emitted on every response (OpenAI Chat Completions,
Anthropic Messages, OpenAI Responses), regardless of whether the
thinking-budget envelope was opted into. Additive to the OpenAI /
Anthropic shape; clients that don't know the field ignore it. For
streaming requests, `timings` appears in the terminal usage chunk
(OpenAI), the `message_delta.usage` event (Anthropic), and the
`response.completed.usage` payload (Responses).

## 7. Close-kind taxonomy

`finish_details.close_kind` records how the `<think>` block ended.
The current taxonomy is:

| Value | Meaning |
|---|---|
| `natural` | The model emitted `</think>` on its own, either before reaching the phase-1 cap or before Level 2 had to force-close. |
| `soft` | The soft-close logit-ratio peek (Level 2.5) fired before the hard cap — `prob[</think>] / prob[chosen_tok]` cleared the operator-configured `soft_close_min_ratio` threshold, and the AR loop injected `</think>` while the model was already "near" closing. Indicates voluntary cooperation: the model would have closed soon anyway; we just hurried it along to reclaim tokens. Currently Qwen3.5/3.6 only. |
| `hard` | The phase-1 cap was reached without a model-emitted `</think>` and without the soft path triggering. Either Level 2 force-closed the block in-loop (preserving KV) or Level 1 ran the phase-2 reprompt. |

When both `soft` and `hard` could fire on the same AR step (the
soft threshold cleared at exactly the budget-edge step), `soft`
wins — the soft trigger carries more information (the model agreed
it was time) than the hard trigger (which only reports coercion).
See `docs/experiments/soft-close-thinking-termination-plan.md` §4 +
§12 for the design rationale.

Soft-close is enabled by the operator via the CLI flag
`--think-soft-close-min-ratio <F>`. Default `0.0` keeps the legacy
two-value taxonomy (`natural` / `hard`); any positive value
activates the third. The dial is a probability ratio in `[0, 1]`:

| `min_ratio` | Behaviour |
|---|---|
| `0.0` | Disabled. Soft path inert; per-request overrides silently ignored. |
| `0.05`–`0.2` | Conservative — fires only when `</think>` is within 5×–20× of the argmax probability. Recommended starting range. |
| `0.5` | Aggressive — fires when `</think>` has at least half the probability of the chosen token. |
| `1.0` | Strict — fires only when `</think>` IS the most-likely token. Useful as a safety check. |

Per-request override (Anthropic envelope, see §4.1):

```jsonc
{
  "thinking": {
    "type": "enabled",
    "soft_close_min_ratio": 0.1
  }
}
```

The per-request value clamps to `min(requested, server_default)` —
clients can tighten (lower the threshold, fire more aggressively)
but not loosen (raise it above the operator's ceiling). When the
server has the dial disabled (`0.0`), per-request overrides are
silently ignored — the feature is operator-policy gated.

## 8. Streaming

Streaming responses (`stream: true`) honor the same configuration
knobs and emit the same reasoning text via the format-appropriate
SSE deltas (OpenAI `delta.reasoning_content`, Anthropic
`content_block_delta` with `thinking_delta`, OpenRouter
`delta.reasoning`).

`finish_details` is emitted in the final chunk for OpenAI Chat and
in the terminal `message_delta` event for Anthropic.

## 9. Out of scope

- **Per-request budget *override* (unclamped).** §4 describes the
  bounded form: clients can request *tighter* budgets than the
  server-configured ceiling, never looser. Allowing full override
  would re-create the silent-truncation footgun of middleboxes that
  drop unknown fields.
- **Spec-decode soft-close peek.** Soft-close fires inside the AR
  loop. When spec-decode is in use, the close still triggers at the
  spec-decode → AR tail-off boundary (slightly later than pure-AR
  mode); the verify/accept inner loop does not run the comparator.
  Gemma 4 and Laguna are pure-AR; this only matters for Qwen3.5/3.6
  with a draft model.
- **Multi-token close joint probability.** When `</think>` tokenizes
  to multiple ids, the soft-close comparator peeks only the FIRST
  id's logit (the existing multi-token inject machinery drives the
  remainder of the sequence on subsequent steps). The joint
  `P(t_0, t_1, …)` peek is left to a v2 if false-positive rates
  warrant it.
- **Per-token close-info metadata.** The upstream reference exposes
  `(token_index, remaining_budget, rank)` for the close event. The
  current `finish_details` reports aggregate counts only.
- **Phase-1/phase-2 split inside `reasoning_details`.** Today the
  list always carries exactly one block. A future version may add
  per-phase blocks (`[{phase:1, …}, {phase:2, …}]`) — the typed-list
  shape was chosen specifically to allow this without breaking
  clients.
