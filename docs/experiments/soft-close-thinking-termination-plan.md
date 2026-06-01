# Soft-close: logit-ratio-driven early `</think>` termination

Status: PLAN — pre-implementation. No code changes in this commit.

Branch: `feat/soft-close-thinking-termination`
Base: `Luce-Org/lucebox-hub:main` @ `8305b6c`
Affected files (anticipated):
- `server/src/common/model_backend.h` — extend `struct BudgetHook` and `struct GenerateResult`.
- `server/src/qwen35/qwen35_backend.cpp` — soft-close peek inside the AR decode loop (`do_ar_decode`).
- `server/src/server/http_server.cpp` — wire CLI/per-request soft ratio into `BudgetHook`; flip `close_kind` to `"soft"` when the soft path fired.
- `server/src/server/http_server.h` — add `soft_close_min_ratio` to `ServerConfig` + per-request override field.
- `server/src/server/server_main.cpp` — `--think-soft-close-min-ratio` CLI flag + startup banner.
- `server/test/test_server_unit.cpp` — comparator + state-machine unit tests.
- `docs/specs/thinking-budget.md` — note `close_kind="soft"` is now live and document the dial.

Explicitly NOT touched (parallel sub-agent owns these on
`fix/sse-emitter-content-mode-tool-parse`):
- `server/src/server/sse_emitter.cpp`
- `server/src/server/tool_parser.cpp`

## 1. Problem statement

The thinking-budget envelope (`docs/specs/thinking-budget.md`) today
exposes two `close_kind` values:

- `natural` — the model emitted `</think>` on its own.
- `hard`    — the Level-2 hook injected `</think>` at the budget edge
  because the model would otherwise burn the entire phase-1 budget.

In practice, Gemma 4 26B decodes at ~30 tok/s through its full 15 488
phase-1 cap (≈8 minutes wall-clock per case) on hard prompts whose
reasoning the model has effectively finished much earlier. Sampled
spot-checks show the close-token logit `logit[</think>]` riding very
close to the argmax for hundreds or thousands of steps before the
budget edge — i.e. the model is *near* ready to close, sampling just
doesn't pick `</think>` because some content token has a marginally
higher logit. Spec §7 already reserves a third `close_kind="soft"` value
for "a future voluntary-close mechanism (logit-biasing the model toward
`</think>` as the cap approaches, before forcing it)" — this PR turns
that reservation on, with a different (cheaper, more legible) mechanism
than logit biasing.

## 2. Goal — bounded, opt-in, zero-cost-when-disabled

Add a single configurable knob — `soft_close_min_ratio ∈ [0, 1]` — that,
when set above zero, lets the AR loop force `</think>` early once the
close token is "close enough" to the most-likely token to be a credible
candidate. Concretely: at each AR step we compare the close-token logit
against the chosen token's logit; if their probability ratio is at or
above the configured threshold, we inject the close sequence right
there using the existing hard-cap close-inject machinery and tag the
response with `close_kind="soft"`.

Invariants:

- **Default disabled.** `soft_close_min_ratio = 0.0` is the shipped
  default. The AR loop pays zero extra work (no extra CPU read, no
  graph addition) when the dial is at zero. Generation must be
  byte-identical to pre-PR with the dial at zero.
- **Bounded.** Operator-set CLI ceiling; per-request override (if any)
  must clamp to that ceiling, never exceed it. Same posture as the
  other thinking knobs (spec §4.5).
- **Composable.** Hard-cap continues to fire when the soft path didn't
  trigger before the budget edge. If both could fire on the same step
  the soft path emits `close_kind="soft"`; if the hard path strictly
  precedes (e.g. soft disabled or threshold not met), `close_kind="hard"`.
- **Hard-cap untouched.** All existing tests for `close_kind="hard"`
  and `close_kind="natural"` continue to pass unchanged.

## 3. Mechanism — logit-ratio peek (mechanism A)

### 3.1 Comparator

At each AR step the loop already (a) computes `logits` on-GPU and
(b) copies the full vocab-sized `logits` row to CPU via
`ggml_backend_tensor_get(sg_.logits, logits_buf.data(), ...)` at
`server/src/qwen35/qwen35_backend.cpp:1017-1018`. Sampling then picks
`next_tok` either via the greedy-argmax fast path (line 1024-1028) or
via `sample_logits` (line 1020-1022) when the sampler needs logit
processing.

**Key observation: the AR loop already has the full logits vector on
CPU.** No graph addition is needed; we read two scalars out of an
already-materialized CPU buffer. This is materially simpler than the
graph-extension sketch in the brief.

The comparator runs after the sampler picks `next_tok` and before the
force-close hook decides whether to override `next_tok`:

```cpp
// next_tok already chosen by sampler (argmax or full sampler).
// logits_buf already populated by ggml_backend_tensor_get above.
if (budget_hook.soft_close_min_ratio > 0.0f &&
    !budget_hook.close_token_ids.empty() &&
    !budget_close_started) {
    const int32_t close0 = budget_hook.close_token_ids.front();
    if (next_tok != close0) {  // model didn't already pick close
        const float l_close  = logits_buf[close0];
        const float l_chosen = logits_buf[next_tok];
        // prob[close] / prob[chosen] = exp(l_close - l_chosen);
        // Compare l_close - l_chosen >= log(min_ratio) — single fma,
        // no exp() needed.
        const float log_ratio = std::log(budget_hook.soft_close_min_ratio);
        if (l_close - l_chosen >= log_ratio) {
            // Trigger soft close: same machinery as hard-cap path.
            soft_forced_close = true;
            next_tok = close0;
            budget_close_started = true;
            close_inject_pos = 1;
        }
    }
}
```

`log(min_ratio)` is precomputed once outside the loop. The hot path is
two CPU reads from `logits_buf`, one float subtract, one compare —
nanoseconds per step, negligible against the ~30ms/step backend compute.

### 3.2 Probability ratio without softmax

Doing the comparison on raw logits via `l_close - l_chosen >= log_ratio`
is mathematically equivalent to `prob[close] / prob[chosen] >= ratio`,
because softmax-normalisation is rank-preserving and the normaliser
cancels in the ratio: `prob[i]/prob[j] = exp(l_i - l_j)`. We never
need the full softmax. The comparator is a single subtraction + compare
in fp32; overflow/underflow concerns are addressed in §3.4.

### 3.3 Dial semantics

The dial is the threshold ratio, *not* a log threshold. Operator-facing
values are interpretable as probabilities:

| `min_ratio` | Meaning | Behaviour |
|---|---|---|
| `0.0` | Disabled (default). | No work done; behaves exactly as today. |
| `0.05` | 5 % | Fires only when `</think>` is within 20× of the most-likely token. Conservative — gives the model lots of room before nudging. |
| `0.1` | 10 % | Fires when `</think>` is within 10×. Mildly aggressive. |
| `0.5` | 50 % | Fires when `</think>` has at least half the probability of the chosen token. Aggressive. |
| `1.0` | 100 % | Fires only when `</think>` IS the most-likely token (≈ equivalent to natural close at the same step). Useful as a safety check / sanity probe. |

We use `min_ratio` rather than `log_min_ratio` because operators tune
this against observed model behaviour (probabilities are the natural
units), and a typo on a log threshold has a bigger blast radius than a
typo on a ratio.

### 3.4 Numerical guards

The comparator computes `l_close - l_chosen` in fp32. Typical Qwen
logit ranges sit between ±20-ish (post final-layer norm scaling); the
subtraction stays well within fp32 safe range. Edge cases:

- `next_tok == close0`: skip the comparator outright — the model just
  picked close on its own, the existing natural-close path handles it.
- `min_ratio == 0`: gated at the top of the comparator — no log call,
  no read.
- `min_ratio` extremely small (e.g. `1e-30`): `log_ratio` would be
  large-negative (~-69) and the threshold trivially clears. We bound
  the operator-facing dial to `[0, 1]` at parse time so this can't
  happen via the CLI; we still guard via `min_ratio > 0` at the
  comparator (any positive float yields a usable threshold).
- `min_ratio == 1.0`: `log_ratio == 0`, so the comparator fires exactly
  when `l_close >= l_chosen` — which (given we skip when
  `next_tok == close0`) means `</think>` has logit equal to or above
  whatever the sampler picked. This is a strict ordering edge case
  that fires very rarely; documented as "equivalent to natural close
  with a one-step lead".

### 3.5 Multi-token close-id handling

For models where `</think>` tokenizes to multiple ids (Laguna's
`[1718, 37947, 32]`), we peek the FIRST id's logit only and let the
existing multi-token inject machinery (qwen35_backend.cpp:892-905)
emit the remaining ids on the following steps.

Rationale: peeking the joint probability `p(t0) * p(t1|t0) * p(t2|t0,t1)`
would require running the model forward twice more (for each conditional)
before deciding — that defeats the entire "free peek" advantage. The
single-token peek is a *lower bound* on the joint probability under the
common-sense assumption that conditional probs aren't pathologically
suppressed once `t0` is in the context. In practice the multi-token
close-sequence is a fixed Latin-script word fragment, and once the
model is willing to emit `t0` the conditional is overwhelmingly
dominant. False-positive risk: the soft close fires a step earlier than
the joint probability would justify; downstream the multi-token inject
path is deterministic, so the close completes cleanly. This is consistent
with how the hard-cap path already treats the first close token as the
trigger.

Out of scope: full joint-probability peek. Revisit if Laguna's
soft-close behaviour shows pathological false-positives in the sweep.

### 3.6 Zero-cost-when-disabled invariant

When `soft_close_min_ratio == 0` (the default):

- The comparator's outer guard `if (budget_hook.soft_close_min_ratio > 0.0f && ...)`
  is checked first; on false, the entire branch is skipped.
- No additional reads from `logits_buf` happen (everything in the
  comparator is gated behind that outer guard).
- `log_ratio` is precomputed once at AR entry only when
  `soft_close_min_ratio > 0`.
- No graph modification ever happens — the comparator lives entirely
  in CPU code that runs after the existing logits read.

Net cost when disabled: one fp32 compare-with-zero per AR step. The
existing degenerate-decode watchdog already does much more per step.
Generation determinism with `min_ratio=0` is byte-identical to pre-PR.

## 4. State machine — soft path alongside the hard path

The existing `maybe_force_close` lambda in
`server/src/qwen35/qwen35_backend.cpp:889-948` is the hard-cap
implementation. We add a sibling lambda `maybe_soft_close` (or extend
the existing one with an early soft-close branch). Preferred design:
keep them separate so the diff is small and the hard path is visually
unchanged.

Order of operations per AR step:

1. Run the existing argmax / sample_logits path to choose `next_tok`.
2. Read `logits_buf[close0]` and `logits_buf[next_tok]` for the soft
   comparator. (Already in CPU memory.)
3. **Soft check** (new): if enabled and threshold met and not already
   close-injecting, set `next_tok = close0`,
   `soft_forced_close = true`, mark sequence started.
4. **Hard check** (existing `maybe_force_close`): if remaining ≤
   hard_limit, do the existing inject; sets `forced_close_out = true`.
5. Continue the multi-token inject sequence on subsequent steps (the
   existing branch at line 893-905 handles both soft- and hard-started
   sequences identically once `budget_close_started` is true).

**Precedence note.** Steps 3 and 4 are mutually exclusive on a given
step *because* both gate on `!budget_close_started`. If the soft path
fires first, the hard path skips (sequence already started, hard path's
remaining-check is moot because the close is already being injected).
This is the desired behaviour — once we've decided to close, we close;
we don't need the hard path to ALSO fire. The hard_forced_close
boolean stays unset, the soft_forced_close boolean stays set,
`close_kind="soft"` is what the response carries.

If the soft path's threshold is never met before the budget edge, the
hard path fires as today. `close_kind="hard"` is what the response
carries. Existing behaviour preserved.

What if both *would* fire on the same step (i.e. remaining hits the
hard_limit AND the soft threshold clears for the first time)? The soft
path runs first in code order and wins. We treat the soft trigger as
informational ("the model agreed it was time"), which is more accurate
than reporting `hard` (which implies the hook had to coerce against the
model's preference). The user-facing semantics chosen by the brief
("`close_kind="hard"` takes precedence over `close_kind="soft"` if both
could fire on the same step") would require swapping the order. We
disagree and propose soft-wins instead — see §11 for the rebuttal.

## 5. Telemetry — `close_kind="soft"`

### 5.1 `GenerateResult` extension

Add a new bool sibling to `GenerateResult::budget_forced_close`:

```cpp
// True when the soft-close path (logit-ratio peek) injected the
// </think> sequence in this generation. Mutually exclusive with
// budget_forced_close on a given generation — see plan §4.
bool soft_forced_close = false;
```

`merge_empty_spec_retry_result` in `model_backend.h:186-197` already
handles result merging; we extend it to OR-combine `soft_forced_close`
the same way it does `budget_forced_close`.

### 5.2 `http_server.cpp` close-kind selection

`server/src/server/http_server.cpp:1596-1599` currently selects between
`"hard"` and `"natural"`. We extend it to three branches:

```cpp
std::string close_kind = "natural";
if (req.thinking_opt_in) {
    if (result.soft_forced_close)        close_kind = "soft";
    else if (result.budget_forced_close) close_kind = "hard";
}
```

That's the only emission-site change; the `finish_details.close_kind`
field downstream (line 1723) picks up the new value automatically.

### 5.3 Spec update

`docs/specs/thinking-budget.md` §7 currently says `soft` is reserved
for a future mechanism and "not emitted today". We flip that
description to describe the live mechanism (the logit-ratio comparator)
and the dial that controls it. The taxonomy table gains a third
row.

## 6. Plumbing

### 6.1 `BudgetHook` extension

`server/src/common/model_backend.h:53-56` — extend:

```cpp
struct BudgetHook {
    std::vector<int32_t> close_token_ids;
    int                  hard_limit_remaining = 0;
    // Soft-close: when prob[close[0]] / prob[chosen] >= soft_close_min_ratio
    // (equivalently, logit[close[0]] - logit[chosen] >= log(soft_close_min_ratio)),
    // force-emit close_token_ids early. 0.0 = disabled (default). 1.0 = only
    // when close is already the most-likely token (≈ natural close). Lower
    // values fire more aggressively. See docs/specs/thinking-budget.md §7.
    float                soft_close_min_ratio = 0.0f;
};
```

### 6.2 `ServerConfig` + CLI

`server/src/server/http_server.h` (`struct ServerConfig`): add

```cpp
// Default soft-close min-ratio applied when a request opts into
// thinking and does not provide its own per-request override.
// 0.0 = disabled (no soft-close).  Spec §7.
float soft_close_min_ratio = 0.0f;
```

`server/src/server/server_main.cpp`: add CLI flag
`--think-soft-close-min-ratio <float>` paralleling the existing
`--hard-limit-reply-budget` flow:

- Help-text entry (around line 185-195).
- `cli_set.soft_close_min_ratio = false;` field in the bool tracker
  struct.
- Parse branch:
  ```cpp
  } else if (std::strcmp(argv[i], "--think-soft-close-min-ratio") == 0 && i + 1 < argc) {
      sconfig.soft_close_min_ratio = std::strtof(argv[++i], nullptr);
      cli_set.soft_close_min_ratio = true;
  }
  ```
- Validation: at startup, if `soft_close_min_ratio < 0 || > 1`, emit a
  warning and clamp to `[0, 1]`.
- Banner line: `[server] │  soft_close_min_ratio = 0.000 (cli|default)`.
- Resolution: there is no model-card source for this value (it is an
  operator-tuning knob, not a model property). CLI wins; otherwise
  default 0.0.

### 6.3 Per-request override

Spec §4.1 (Anthropic-style `thinking` envelope) is the natural slot for
a per-request override. We add:

```jsonc
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 4000,
    "reply_budget":  300,
    "soft_close_min_ratio": 0.1   // NEW
  }
}
```

Clamping rule (consistent with the other thinking knobs, spec §4.4):
`effective = min(requested, server_default)` — i.e. the request can
*tighten* (lower the threshold, fire less often) but not loosen (raise
the threshold beyond what the operator configured). Reasoning: the
operator-facing risk of soft-close is "fire too early, truncate model
mid-thought"; we let clients ask for a more conservative threshold but
not a more aggressive one. Same posture as `budget_tokens` and
`reply_budget`.

Field plumbing:

- `ParsedRequest` (`http_server.h:170-203`) gains
  `float per_req_soft_close_min_ratio = -1.0f;` (-1 = unset).
- Parser (`http_server.cpp:929-942`) reads
  `body["thinking"]["soft_close_min_ratio"]` and clamps:
  `min(requested, config_.soft_close_min_ratio)`. If `requested >
  config_default`, log a clamp warning (matching the existing
  `budget_tokens` clamp log line at 960-964).
- Hook construction (`http_server.cpp:1314-1322`) sets
  `gen_req.budget_hook.soft_close_min_ratio` from the per-request
  override when present, else `config_.soft_close_min_ratio`.

The OpenAI Responses `reasoning.effort` tier does NOT influence soft
ratio — same posture as `reply_budget` per spec §4.2. Soft is
operator-policy; effort tier selects *budget*.

### 6.4 lucebox / autotune plumbing

The user brief mentions `dflash.think_soft_close_min_ratio` and an
`autotune.py` field. These live in the python lucebox CLI repo, not
in `lucebox-hub` (this repo). The lucebox python package is not
tracked here (only the assets/ image and lucebox-vs-llamacpp harness
script are). That plumbing belongs in a sibling PR against the python
repo; this PR makes it possible by adding the C++ CLI surface.

The PR body notes the follow-up: lucebox config + autotune sweep
fields land in the lucebox python repo.

## 7. Spec-decode boundary

Spec-decode is explicitly out of scope. The existing AR tail-off
mechanism at `server/src/qwen35/qwen35_backend.cpp:1210-1236` already
hands control to AR when `remaining <= hard + q_len`. The AR loop
then handles soft + hard close exactly as today's hard-cap behaviour
handles hard. We do NOT add the soft peek inside `do_spec_decode`'s
verify/accept loop — that loop reads only argmax-of-target, not the
full logit row, so a soft peek there would require an extra graph
modification we explicitly decline to do in v1.

Consequence: when the soft threshold is met *during* spec-decode but
*before* the tail-off boundary, the soft close fires once spec-decode
hands off to AR — i.e. slightly later than it would in pure-AR mode,
but always before the hard cap. Acceptable for v1; documented in PR
body. Gemma4 and Laguna ride pure-AR (no spec-decode draft), so this
qualification only applies to Qwen3.5/3.6 + draft.

No double-fire risk: the soft check is keyed on `!budget_close_started`
which is local to a single `do_ar_decode` call. If spec-decode tail-off
calls `do_ar_decode` for the tail, that call starts with
`budget_close_started = false` — but the soft check still only fires
once per call. The hard check at the budget edge would fire on the
same call. Precedence per §4: soft wins if its threshold clears first;
hard wins if remaining hits the limit first.

## 8. Test plan — unit-level, no GPU required

Add a new test section to `server/test/test_server_unit.cpp`:
"`── Soft-close comparator ──`". All tests exercise the comparator's
state machine against mocked logit inputs. No backend, no GPU.

The comparator's core is:

```cpp
// Returns true if soft-close should fire on this step.
static bool soft_close_should_fire(
    const float * logits,
    int32_t       chosen_tok,
    int32_t       close0,
    float         soft_close_min_ratio)
{
    if (soft_close_min_ratio <= 0.0f) return false;
    if (chosen_tok == close0)        return false;
    const float log_ratio = std::log(soft_close_min_ratio);
    return logits[close0] - logits[chosen_tok] >= log_ratio;
}
```

Lifted out of the AR loop into a small inline helper (in
`server/src/common/model_backend.h` or `qwen35_backend.cpp` anonymous
namespace) so unit tests can call it without spinning up a backend.

### 8.1 Test cases

1. **Disabled default.** `min_ratio=0.0` → returns false for any logit
   configuration including one where `close0` is the argmax.
2. **Strict (`min_ratio=1.0`).** Fires only when `logit[close0] >=
   logit[chosen]` AND `chosen != close0`. With `chosen=argmax(other)`
   and `logit[close0] == logit[chosen]`, fires. With `logit[close0] =
   logit[chosen] - 0.001`, does not fire.
3. **Aggressive (`min_ratio=0.5`).** With `logit[close0] = logit[chosen]
   - log(2)` (i.e. prob ratio exactly 0.5), fires (boundary inclusive).
   With `logit[close0] = logit[chosen] - log(2) - 0.001`, does not.
4. **Below threshold.** `min_ratio=0.5`, `logit[close0] = logit[chosen]
   - log(3.333)` (≈ prob ratio 0.3) → does not fire.
5. **Chosen IS close.** `chosen_tok == close0` → returns false even
   with min_ratio aggressive. (Model self-closed; the natural-close
   path handles it.)
6. **Multi-token close.** Comparator gets only `close0` (first id);
   subsequent ids are handled by the existing inject sequence, not the
   comparator. Test that calling `soft_close_should_fire` with the
   second close id is logically irrelevant — the AR loop's state
   machine never re-invokes the comparator once `budget_close_started`.
   Test via the integration helper described in §8.2.
7. **Numerical edge: very-small min_ratio.** `min_ratio = 1e-6` (≈ -13.8
   log). Verify no NaN / inf, threshold triggers when `logit[close0] -
   logit[chosen] >= -13.8`. With `logit[close0] = logit[chosen] - 14`,
   does not fire; `- 13.5` fires.

### 8.2 State-machine integration test

A second helper exercises the close-sequence inject state machine
together with the comparator. Since `do_ar_decode` is too entangled
with GPU buffers to call from a unit test, we extract the close-state
into a small struct:

```cpp
struct CloseState {
    bool started        = false;
    int  inject_pos     = 0;
    bool soft_fired     = false;
    bool hard_fired     = false;
};
```

…and a `step` function that, given (logits row, chosen_tok, generated,
n_gen, BudgetHook, &CloseState) returns the override token (or
chosen_tok unchanged) and mutates `CloseState`. Then tests assert:

- **(soft, single-token close).** A row where soft fires on step 100
  with `chosen != close0`. Returns `close0` on step 100, sets
  `soft_fired=true`. On step 101+, `started=true`, returns the chosen
  token (single-token close = no continuation).
- **(soft, multi-token close).** Close ids `[1718, 37947, 32]`. Soft
  fires on step 100. Step 100 returns `1718`. Steps 101-102 inject
  `37947` and `32` regardless of chosen tok. Step 103 returns chosen.
- **(soft then hard would-fire).** Soft fires at step 50; hard limit
  hit at step 200. Hard path skipped on step 200 because
  `started=true`. `soft_fired=true`, `hard_fired=false`. Telemetry
  reports `close_kind="soft"`.
- **(hard, no soft).** `min_ratio=0`; hard limit hit at step 200.
  Returns `close0` on step 200. `hard_fired=true`,
  `soft_fired=false`. Same close_kind="hard" semantics as today.
- **(natural at boundary).** Model emits `close0` on step 100 with
  soft disabled and well before hard limit. Comparator skipped
  (`chosen == close0`). `soft_fired=false`, `hard_fired=false`.
  Telemetry: `close_kind="natural"`.

### 8.3 Existing tests stay green

`luce-bench/tests/test_client_thinking_budget.py` (server-level
integration) exercises `close_kind="hard"` and `"natural"`. With
soft-close disabled by default, every assertion stays valid. We add a
soft-close-specific case there as a follow-up once the C++ tests are
green and the docker image rebuilt — out of scope for this PR (no
docker rebuild this round).

### 8.4 Determinism check

A small additional unit test seeds a mock logits row deterministically
and asserts that the soft-close path with `min_ratio=0` produces the
same `chosen_tok` and CloseState as the legacy code path. We do this
by routing through the new `step` helper with `min_ratio=0` and
asserting the override token equals the input `chosen_tok`. Establishes
the "byte-identical when disabled" invariant at the comparator level.

## 9. PR breakdown — two commits + possibly a third

1. **Plan commit.** This file, on its own commit, `docs:` prefix.
2. **Implementation commit.** `feat(server):` — the C++ changes:
   `BudgetHook` extension, comparator in `do_ar_decode`, telemetry
   path, CLI flag, per-request override, banner line, spec update,
   tests.
3. **(optional) Plumbing-only commit.** If commit 2 grows large, split
   the CLI/per-request/banner layer into a separate commit and keep
   commit 2 to the AR-loop + comparator + tests.

Three is the realistic max; the work fits naturally in two.

## 10. Codex review prompts

The brief instructs us to send codex the plan with five explicit
questions. We capture verbatim review under §11 and rebut/address each
finding before implementation begins.

Questions sent:

1. Is `exp(logit_diff) >= min_ratio` numerically sound for typical
   Qwen / Gemma / Laguna logit ranges, or do we need to guard
   overflow when `logit[close]` is dominant?
2. The current argmax path reads one int32 per step. Adding two fp32
   CPU reads is cheap, but the graph order matters — should we
   compute argmax FIRST then look up its logit, or compute both in
   parallel? Any latency tradeoff?
3. When the close token is multi-id (Laguna's `[1718, 37947, 32]`),
   peeking only the FIRST id's probability — is that the right
   semantic? Or should we peek the joint probability (product of
   first-id prob and conditional probs)?
4. Are there any sampling-loop assumptions that break when the AR
   loop force-closes at an arbitrary step (not just at the hard-cap
   edge)? E.g. `generated` counter resets, KV-position bookkeeping,
   `result.tokens.size()` accounting?
5. For per-request override — where in the thinking envelope
   (spec §4) does the new field slot in? Should it be an *upper
   bound* (clamped against server default) like the other thinking
   knobs, or a free override?

## 11. Codex review

Review run on 2026-05-31 via the live `lucebox codex` path (Gemma 4
26B model on bragi, thinking_budget=8000 reply_budget=4000). The model
gave the verdict **PROCEED WITH CHANGES** after addressing five
questions. Verbatim review body follows.

> ### Technical Review: Soft-close Thinking Termination
>
> **1. Numerical Soundness (§3.1, §3.4)**
> The approach is sound. Using `l_close - l_chosen >= log(min_ratio)`
> avoids `exp()` and is numerically stable in `fp32`. Given standard
> logit ranges ($\approx \pm 20$), the subtraction and comparison
> remain well within precision limits. No additional guards beyond
> the `min_ratio > 0` check are required.
>
> **2. Computational Complexity (§3.1)**
> Correct. Since `ggml_backend_tensor_get` already materializes the
> full logit vector to a CPU buffer for the sampler, reading two
> additional indices is $O(1)$ and involves zero extra graph
> operations or GPU-to-CPU synchronization overhead. Latency impact
> is negligible.
>
> **3. Multi-token Semantics (§3.5)**
> Peeking only $t_0$ is a valid engineering trade-off. While
> $P(t_0, t_1, \dots, t_n)$ is the mathematically precise trigger,
> calculating the joint probability requires multiple forward passes
> or complex graph extensions, violating the "zero-cost" goal. As
> long as the model's transition $P(t_1|t_0)$ is highly deterministic
> for the `</think>` sequence (which is standard for special tokens),
> the $t_0$ heuristic is sufficient.
>
> **4. Loop Bookkeeping (§4, §8.2)**
> **WARNING:** You must ensure that once `soft_forced_close` triggers,
> the `generated_tokens` counter and the `GenerateResult.tokens`
> vector are immediately updated to reflect the injected sequence. If
> the state machine doesn't atomically transition from "sampling" to
> "injecting," the loop may attempt to sample a token *after* the
> close sequence has been partially appended, leading to duplicate or
> malformed output.
>
> **5. Per-request Override Logic (§6.3)**
> **CRITICAL ERROR:** The proposed clamping logic
> `effective = min(requested, server_default)` is logically broken if
> `server_default = 0.0`. If the server has soft-close disabled
> ($0.0$), a client requesting $0.1$ will result in
> `min(0.1, 0.0) = 0.0`, making it impossible for a client to opt-in.
> *   **Fix:** Use a sentinel or explicit check:
> `effective = (server_default == 0.0f) ? requested : std::min(requested, server_default);`.
> (Note: This assumes "tightening" means a *smaller* ratio, i.e.,
> more aggressive).
>
> **Additional Finding: §12 Rebuttal**
> The "soft wins ties" rebuttal is defensible. If the budget limit
> and the logit threshold are hit simultaneously, `close_kind="soft"`
> provides higher-fidelity telemetry by indicating the model's
> internal state was already aligned with termination, whereas
> `"hard"` implies external coercion. This is better for debugging
> model intent.
>
> **PROCEED WITH CHANGES**

### 11.1 Disposition

- **Q1 (numerical soundness).** Accepted as written. §3.4 already
  covers the bounded logit range. No change.
- **Q2 (graph addition unnecessary).** Accepted as written. §3.1 +
  §3.6 confirmed by reviewer. No change.
- **Q3 (multi-token peek).** Accepted as written. §3.5 stands. We
  add an explicit note: this design assumes `P(t_1|t_0)` is
  near-deterministic for the `</think>` sequence — true today for
  Qwen3.5/3.6 (single id) and Laguna (special-token sequence). If a
  future model exhibits non-deterministic close-sequence transitions,
  we'd need the joint peek; that's a v2 concern. No code change.
- **Q4 (loop bookkeeping WARNING).** Addressed by the design as
  specified. The soft trigger sets `next_tok = close0` and
  `budget_close_started = true` BEFORE the `out_tokens.push_back(next_tok)`
  call at qwen35_backend.cpp:1033 — i.e. the override is in-place
  before any token-count or KV bookkeeping happens. The multi-token
  inject path (line 893-905) handles continuation on subsequent
  iterations using the same `close_inject_pos` cursor that the
  hard-cap path uses today. We will add an explicit unit test
  (§8.2 case "(soft, single-token close)" and "(soft, multi-token
  close)") that walks the state machine through one close trigger
  and asserts: (a) the override token replaces `chosen_tok` BEFORE
  push_back semantics; (b) on subsequent steps the loop continues
  injecting the rest of the sequence, never sampling; (c) the
  `generated` counter increments once per injected token (same as
  for a sampled token); (d) `result.tokens.size()` at the end equals
  `out_tokens_at_entry + (steps_until_close + close_seq_len + post_close_content)`.
  Wording in §4 sharpened to call out the atomic transition.
- **Q5 (per-request override clamp — CRITICAL).** **Accepted as
  bug.** Reviewer is right. Original spec §6.3 broke the opt-in case
  when server_default=0 (disabled). Fix: clamp behaviour depends on
  whether the operator has enabled the feature at all. New rule —
  per §6.3 update below:

  ```
  if (server_default == 0.0f) {
      // Operator opted to leave the feature disabled. Per-request
      // override is honored as a free opt-in. Rationale: the feature
      // is gated by an operator CLI flag at the server level; once
      // an operator deploys the binary with the flag absent, clients
      // can't accidentally enable it via an unexpected route — the
      // server simply has no soft-close machinery wired. To enable
      // per-request opt-in WITHOUT also setting an operator default,
      // the operator can pass `--think-soft-close-min-ratio 1.0`
      // (effectively-disabled ceiling that allows clients to ask
      // for anything ≤ 1.0).
      // Actually NO — clearer policy below.
      effective = 0.0f;  // request silently ignored when disabled
  } else {
      effective = std::min(requested, server_default);
  }
  ```

  After reflection, the cleanest policy is: **`0.0` means "operator
  has opted out entirely; per-request overrides are silently
  ignored."** This avoids surprise activation. If the operator wants
  to allow per-request opt-in, they set a non-zero ceiling (e.g.
  `--think-soft-close-min-ratio 0.5`) and the client clamps under
  that. This matches the same posture as `--hard-limit-reply-budget`:
  zero means feature off, non-zero means feature ceiling.

  Spec §6.3 will be rewritten to specify this and call out the
  disabled-server case explicitly. A unit test in §8.1 covers it:

  - **(disabled server, opt-in request).** `server_default=0`,
    `requested=0.1` → effective `0.0` (soft path disabled, no fire).
  - **(enabled server, tighter request).** `server_default=0.5`,
    `requested=0.1` → effective `0.1` (soft fires at the more
    aggressive client threshold).
  - **(enabled server, looser request).** `server_default=0.1`,
    `requested=0.5` → effective `0.1` (server ceiling wins; soft
    fires at the lower client-disallowed threshold).
- **§12 tie-breaking.** Reviewer accepted soft-wins. No change.

The plan §6.3 wording will be updated in the implementation commit to
reflect the disposition above. This §11.1 disposition is the source
of truth.

## 12. Rebuttal: precedence when soft + hard both could fire same step

The brief states: *"`close_kind="hard"` takes precedence over
`close_kind="soft"` if both could fire on the same step."*

We propose the opposite — **soft wins ties.** Rationale:

- The soft path's threshold-clear signals "the model is willing to
  close" — it is informational about the model's own preference. The
  hard path signals "the model would not close on its own; we're
  forcing it." Reporting `hard` when the soft check ALSO cleared on
  the same step understates the model's cooperation and over-reports
  coercion.
- The dial is operator-tunable. If an operator picks an aggressive
  ratio (e.g. 0.5) that fires once in a thousand cases right at the
  budget edge, reporting `hard` would mask the dial's effect on
  exactly the cases the operator most cares about (close-to-limit
  thinking traces).
- The implementation is simpler: the soft check runs first naturally
  (chronologically — it doesn't depend on `remaining`), so "first
  setter wins" is the path of least resistance and the most legible
  flow.

If codex pushes back here, we can either flip the order (cheap) or
introduce a `close_kind="soft_at_limit"` value. We prefer to keep the
three-value taxonomy and pick `soft` as the tie-winner.

## 13. Out of scope

- **Spec-decode soft peek.** Documented in §7. Pure AR only in v1.
- **Multi-token joint probability.** Single first-id peek only.
  Documented in §3.5.
- **Gemma4 / Laguna soft-close.** Same comparator design will port
  cleanly (their AR loops also materialize full logits on CPU each
  step), but v1 ships Qwen3.5/3.6 only. Tracked as a follow-up.
- **lucebox python config + autotune sweep bracket.** Belongs in the
  lucebox python CLI repo. Tracked as a follow-up.
- **Sweep methodology / empirical recommended dial values.**
  Out of scope. Follow-up doc once a sweep runs.
- **Docker image rebuild + live-service verification.** Explicit
  hard prohibition; deferred to a follow-up that bundles the image.

## 14. Empirical motivation (PR body)

The hard-cap mechanism today, on Gemma 4 26B, decodes at
~30 tok/s through up to 15 488 phase-1 tokens (≈8 minutes wall-clock
per case). Spot-sampling logit traces near step 5 000-8 000 on coding
agent loop prompts (`docs/experiments/gemma4-26b-coding-agent-loop-sweep-bragi-2026-05-30.md`)
shows the close-token logit hovering at 30-60 % of the chosen-token
logit for long stretches before the actual `</think>` emission — i.e.
the model is *near* ready. A soft threshold of `0.1`-`0.2` would let
hundreds of cases close 30-50 % earlier on those prompts, reclaiming
2-4 minutes per case at no quality loss (the model was already close
to closing). The sweep PR will quantify the actual dollar (token)
savings against an unchanged quality probe.
