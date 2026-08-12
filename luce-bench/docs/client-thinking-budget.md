# Design: bounding thinking on budget-unmanaged backends

Status: proposal (revised after codex review) · Owner: luce-bench · Default: off

## Problem

luce-bench sends one think-mode request shape to every backend (`max_tokens`,
`chat_template_kwargs.enable_thinking`, `thinking:{type:enabled}`,
`reasoning_effort:high`). Backends that manage the thinking budget server-side
(lucebox: count tokens, force `</think>` before the cap, reserve reply room)
keep the answer intact. Backends that don't (some OpenRouter routes, mlx_lm)
reason to the `max_tokens` ceiling and truncate the answer, scoring far below
nothink. Measured: OpenRouter qwen3.6-27b think ds4-eval 48.9% vs 72.8% nothink,
32/92 rows length-capped.

`max_tokens` is a total ceiling, so lowering it makes truncation worse. There is
no client-side way to bound thinking on a single stateless request.

## Foundation: model-card resolution in luce-bench (build this first)

Everything below is wrong unless the client uses the *right* tokens for the
*right* model. Today it doesn't: `_thinking.py` carries a hardcoded `FAMILY_TOKENS`
map (Qwen-only, `/think`/`/no_think` only); `runner.py` accepts a `model_card` but
luce-bench has no card registry, so a card only arrives from `/props.model_card`
on a lucebox server. Against OpenRouter/MLX it falls back to the Qwen guess and
knows nothing about terminators, reply reserves, effort tiers, or other families.

Fix: give luce-bench a **card registry** resolvable by model id, mirroring the
cards in `share/model_cards/`. The server already proved the budget work is
card-driven; we duplicate the minimum of that on the client so it works without
`/props`.

Resolution order per run:
1. `/props.model_card` — authoritative when present (lucebox). No preflight
   needed; the server is telling us the exact card it loaded.
2. luce-bench bundled card registry, keyed by the server's
   `normalize_model_card_stem` logic (so `qwen/qwen3.6-27b:free` → `qwen3.6-27b`).
   Bundled as package data so the standalone PyPI build has them.
3. Family fallback (today's `FAMILY_TOKENS`) — last resort, tokens only, logged
   as low-confidence.

**A bundled-card id match is a HINT, not proof.** OpenRouter-style ids, aliases,
quant routes, and finetunes can normalize to `qwen3.6-27b` while serving behavior
that disagrees with the card. So for any source other than `/props.model_card`,
activation of think/nothink control or client abort is **gated on per
provider+route+model preflight success** (see Gating preflight) — not on the id
match alone. The match selects *which* card to test; the preflight decides whether
to *use* it.

Normalization must be pinned by tests, not described: a shared test corpus
covering `qwen/qwen3.6-27b:free`, provider prefixes, version/quant suffixes,
aliases, and unknown revisions — luce-bench either shares the server's
implementation or is tested against the same corpus.

Provenance, recorded per row: `card_source` (`props`|`bundled`|`family`|`none`),
`card_stem`, `card_hash`. When `/props` and a same-stem bundled card disagree,
warn and record both, so stale embedded cards and server/client skew are
auditable rather than silent.

The card supplies, per model, exactly these and the feature reads them rather
than hardcoding:
- `thinking_control.{think_prompt_token, nothink_prompt_token, injection_point}`
  — in-band mode control (Qwen: `/think`/`/no_think` at `user_turn_suffix`).
- `thinking_terminator_hint` — the force-close phrase (Qwen's sentence; Gemma's
  differs). Any client-side termination uses THIS, never a hardcoded `</think>`.
- `thinking_marker` — the channel delimiter for detecting end-of-thinking in the
  stream. (Note: `qwen3.6-27b.json` currently has this null — a real gap; client
  stream-termination needs it populated, or it falls back to `<think>` tags if
  present; genuinely unmarked output stays unsupported, never guessed.)
- `hard_limit_reply_budget` — reply reserve (4096).
- `reasoning_effort_tiers` — the budget values (low 4032 / medium 16128 / …).

**Capability gate:** think-mode control and budgeting activate **only** for a
model that resolves to a thinking-capable card (has a thinking channel + the
fields above). For anything else, record `card="none"` / `not_thinking_capable`
and run plain — never inject tokens into a model that has no thinking channel.

## Avoiding card duplication

Canonical source is `share/model_cards/*.json` + `_schema.json`, read from disk by
the C++ server (`…/share/model_cards/<stem>.json`; `/opt/lucebox-hub/share/...` in
the image). That on-disk path is the server's contract and stays put.

Duplication can only appear in **one** place: luce-bench published standalone to
PyPI, where it can't reach repo `share/`. In-repo and in-container, every consumer
(C++ server, `lucebox` CLI, luce-bench as a workspace member) already reads the
same `share/model_cards/` — no copy.

Resolution:
- No hand-maintained second copy under `luce-bench/`.
- luce-bench **embeds the cards at build time** (hatch `force-include` / build
  hook copies `share/model_cards/*.json` + `_schema.json` into the wheel/sdist as
  `lucebench/_model_cards/`). Git tree carries no duplicate; only built artifacts
  do.
- Runtime prefers `/props.model_card` (always current); the embedded set is the
  no-`/props` fallback only.
- CI drift guard: CI **builds the wheel/sdist** and compares the *packaged*
  card resources' hashes to `share/model_cards/`. (Since the embedded files
  aren't committed, hashing the source tree proves nothing — the guard must
  inspect the built artifact.) Drift fails the build.
- Not a shared pip `lucebox-cards` package: the C++ server can't pip-install it,
  so it would not unify consumers, only add a third artifact.

## Two tiers, cheapest first

### Tier 1 (preferred): native budget hints

Before any client machinery, test whether a backend honors a budget hint it
already understands:

- `reasoning_effort: low | medium` instead of `high`.
- Anthropic-shape `thinking: {type: enabled, budget_tokens: N}` (luce-bench
  already sends this shape for the `probe` area).

If a provider respects either, we get budgeted think with **zero client code**
(one request, one transcript — far cleaner than Tier 2). It is **not** free of
comparability concerns, though: `reasoning_effort: medium` is a different
benchmark setting from default/high think, so it is reported as its **own mode**
(`mode="native_effort"` / `"native_budget"`) and not pooled with plain think.
This is the default path the feature should pursue. luce-bench would expose
`--reasoning-effort {low,medium,high}` and/or `--thinking-budget-tokens N` that
set the request fields; the existing post-run verifier already records whether
the returned reasoning-token count actually fell.

### Tier 2 (fallback, separate experimental mode): client abort + re-prompt

Only for backends that (a) mark reasoning in the stream, (b) accept an
assistant-prefill continuation, and (c) ignore the Tier-1 native knobs. This set
may be small or empty; **gate building it on the preflight below.**

Mechanism, framed honestly:
1. **Count** reasoning tokens as the stream arrives (see Counting).
2. **Stop** consuming and close the connection when over budget. Note: closing
   the HTTP stream does **not** reliably stop server-side generation, billing, or
   load. We stop *reading*; the server may keep going.
3. **Re-prompt** with a second, independent request that re-conditions on the
   captured partial reasoning plus a terminator (card `thinking_terminator_hint`,
   else `</think>\n\n`) as an assistant-prefill turn, think disabled, with its
   own reply `max_tokens` (default 4096). This is a **fresh conditioned sample,
   not a resumption of the original decode.** The answer it produces is what we
   grade.

## Gating preflight (do this before building Tier 2)

Probe each candidate backend and record a capability matrix:

| capability | how to detect |
|---|---|
| native effort honored | send `reasoning_effort:medium` on N≥5 probe prompts; median reasoning tokens drop materially (e.g. ≥30%) vs `high` |
| native budget_tokens honored | send `thinking.budget_tokens=B`; reasoning tokens stay ≤ ~B on ≥80% of probes |
| reasoning marked in stream | `reasoning_content` deltas present, or `<think>` tags in content |
| assistant-prefill accepted | continuation returns a clean answer, not an error/empty |
| usage after abort | does a closed stream still yield usage |

If a backend passes Tier 1, Tier 2 is unnecessary for it. If no backend needs
Tier 2 (Tier 1 covers them, or they can't support the continuation), do not build
Tier 2; classify the unsupported backends instead.

## Identifiability (Tier 2 hard constraint)

Client termination only works when thinking is distinguishable in the stream:

- **`reasoning_content` deltas** present → count those. Best case.
- **`<think>…</think>` in `content`** → detect open tag, count until close.
- **Unmarked** (reasoning indistinguishable from answer, no tags — the lucebox
  /sindri server bug) → no boundary exists; do not guess. Record
  `marking="unmarked"`, run normally, grade as-is. This feature cannot and must
  not try to fix unmarked output; that is a server-side issue.

## Counting

Final usage only arrives on the last chunk, so mid-stream the budget is checked
against an **approximate** reasoning-token estimate (char/4 over accumulated
reasoning text; no tokenizer dependency). Documented caveats: it miscounts
math/code/CJK/whitespace-heavy text, and it overshoots the budget by up to one
streamed chunk before the abort fires. The budget is a soft gate, not an exact
cutoff.

## Comparability and result schema (the part that keeps numbers honest)

Tier 2 is a **distinct benchmark mode**. Its scores are NOT pooled with, and NOT
compared against, single-pass think or lucebox server-managed think. The report
puts budgeted-mode runs in their own bucket with a visible label.

Per-row block `client_thinking`:
- `mode`: "native_effort" | "native_budget" | "client_abort" | "off"
- `requested_budget`: int | null
- `engaged`: bool
- `marking`: "reasoning_content" | "think_tags" | "unmarked"
- `answer_started_before_abort`: bool (if visible content had already begun, a
  re-prompt can duplicate/corrupt the answer; flag it)
- `reasoning_tokens_at_abort`: int (estimate)
- `continuation`: "ok" | "unsupported" | "skipped"

Token/latency accounting is reported as separate fields, never summed into a
single comparable number: original observed tokens (may lack final usage on an
aborted stream), estimated reasoning-at-abort, and continuation usage. Total
cost/latency for Tier 2 is marked not-comparable to single-pass.

Continuation-failure rows (`continuation="unsupported"`) and
`answer_started_before_abort=true` rows are broken out and excluded from
budgeted-mode aggregate accuracy, so a provider-capability failure or a corrupted
re-prompt is never conflated with a model over-thinking failure. Crucially, the
**headline always shows coverage (the denominator)** alongside the score — a route
with many excluded rows must not be able to look artificially strong on a shrunken
sample.

## CLI

- `--reasoning-effort {low,medium,high}` (Tier 1)
- `--thinking-budget-tokens N` (Tier 1, Anthropic shape)
- `--client-thinking-budget N` (Tier 2, opt-in, default unset/off)

All meaningful only with `--think`; no-ops in nothink. Tier 2 additionally
requires the backend to pass the preflight or it degrades to a recorded no-op.

## Non-goals

- Not a replacement for lucebox server-side force-close (single pass, no extra
  round-trip, better).
- **Not a fix for the lucebox/sindri unmarked-reasoning bug.** That is fixed
  server-side by emitting `reasoning_content` / `</think>`. Until then this
  feature reports `marking="unmarked"` and does nothing.
- Not on by default; no behavior change unless a flag is set.

## Test plan

- Unit: native effort/budget flags set the right request fields; nothink no-ops.
- Unit: `reasoning_content`-marked stream over budget → Tier 2 abort fires,
  re-prompt built with terminator + prefill; `mode="client_abort"`.
- Unit: `<think>`-tag stream → boundary detected at `</think>`.
- Unit: unmarked stream → no abort, `marking="unmarked"`, normal grade.
- Unit: continuation returns empty/errors → `continuation="unsupported"`, row
  excluded from budgeted aggregate, no exception.
- Unit: budget not hit → byte-identical to current single-request behavior.
- Unit: report refuses to pool client_abort rows with single-pass rows.

## Open questions

- Estimate vs exact token count: keep char/4, or load the model tokenizer if
  budgets prove imprecise.
- Multi-turn areas (agentic-session): out of scope v1.
- Under `--parallel`, Tier 2 adds a second request per terminated row (load, not
  correctness).
