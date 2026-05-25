# Thinking-control probe protocol

A reproducible 5-mode probe that characterizes how a `/v1/chat/completions`
server handles reasoning/thinking on a single ds4-eval case. Sister of
`bench_http_capability.py --area ds4-eval`, but narrower: one case, one
model, five carefully-chosen request shapes.

The point is to answer four questions per (server, model) pair:

1. **Thinking ON**: does the model produce a *separate* reasoning block,
   properly terminated, that the server correctly routes to
   `message.reasoning_content` rather than leaking into `message.content`?
2. **Thinking OFF**: does the model *actually* skip reasoning, or does
   it do the same work and just hide the tags?
3. **Budget control via server**: does the server's
   soft/hard force-close at `budget_tokens` work, or does thinking flow
   past the close into the visible answer?
4. **Budget control via prompt**: can we control thinking by
   manipulating the system prompt or the rendered template, in cases
   where the server's mechanism fails?

## The five modes

| Mode | thinking | enable_thinking | sys prompt | budget_tokens |
|---|---|---|---|---|
| `think-default`       | enabled  | true  | default | none |
| `nothink`             | disabled | false | default | none |
| `think-low`           | enabled  | true  | default | 1024 |
| `think-medium`        | enabled  | true  | default | 4096 |
| `think-raw-noprompt`  | enabled  | false | empty   | none |

`enable_thinking` is the Jinja flag servers pass into the chat-template
render; for Gemma 4 it controls whether `<|think|>` is emitted in the
system turn. `thinking` is the Anthropic-shape opt-in our server reads
for the budget envelope.

`think-raw-noprompt` is a "naked" mode that combines a contradictory
request (server says think, template says don't, system message empty)
to probe whether the model self-initiates reasoning even without any
prompt-side encouragement.

## What we capture

For every mode the runner saves a per-mode JSON with the full request
envelope, full response envelope, and a flat `row` containing:

* `content_len_chars`, `reasoning_len_chars` — quick "did reasoning go
  to the right field" check.
* `prompt_tokens`, `completion_tokens`, `thinking_tokens` — token-level
  view of where the budget went. `thinking_tokens` comes from
  `usage.thinking_tokens` if the server emits it, otherwise from
  `reasoning_tokens` (Anthropic shape).
* `finish_reason` + `finish_details` — distinguishes `stop` (model
  emitted close-token cleanly) from `length` (max_tokens hit) from
  `hard_close` (server force-closed at budget).
* `prefill_ms`, `decode_ms`, `decode_tokens_per_sec` — timings; not
  the primary signal but useful for cross-mode wall-time comparison.

A combined `_summary.json` + markdown table lands in the snapshot dir,
plus a `_run.log` of the runner's stdout.

## How to run

```bash
# Standard: probe aime2025-02 on whatever's at :8080
SNAPDIR=dflash/docs/tuning-snapshots/<host>-<model-tag>-thinking-control-$(date -u +%Y-%m-%d)
python dflash/scripts/probe_thinking_control.py \
  --url http://localhost:8080 \
  --model dflash \
  --case-id aime2025-02 \
  --out-dir "$SNAPDIR"

# Subset of modes (e.g. skip the long ones)
python dflash/scripts/probe_thinking_control.py \
  ... --modes think-default,nothink
```

Default case is `aime2025-02` (geometry, answer=588) — a hard reasoning
problem that should obviously benefit from a thinking budget but is
short enough to not blow past 8k tokens of decode on most models.

## How to interpret

For each (server, model) tuple, answer these in the writeup:

**Question 1 — thinking ON works correctly?**
* Look at `think-default`. Is `reasoning_content` non-empty?
  → Yes: server's reasoning parser is at least firing.
* Compare `reasoning_len_chars` to `content_len_chars`. Reasoning
  should dominate for AIME (typically 5-50× larger).
* Does `content` contain `<think>` / `<|channel>thought` / `<channel|>`
  literal substrings? Any leakage means the parser isn't catching the
  close token.
* `finish_reason` should be `stop`, not `length` or `hard_close`.

**Question 2 — thinking OFF actually saves work?**
* Compare `nothink.completion_tokens` to `think-default.completion_tokens`.
  Drop of ≥80% = real skip. Drop of <30% = "just hiding the tags".
* `nothink.reasoning_content` must be empty.
* `nothink.wall_s` should drop proportionally to `completion_tokens`
  (modulo prefill overhead).

**Question 3 — budget control via server**
* `think-low` should hit a `finish_details.hard_close` (or similar
  forced-close marker) and have `thinking_tokens ≤ 1024 + a small
  slack`. If `thinking_tokens >> 1024`, the budget was ignored.
* If the forced-close fires but `content` contains residual reasoning
  text, the server closed the *reasoning* block but the model kept
  reasoning in the visible answer.

**Question 4 — prompt-only control**
* `think-raw-noprompt` is a stress test: request asks for thinking,
  template says no, prompt is empty. Whether the model self-thinks
  reveals how much thinking is intrinsic to the weights vs prompted.

## When to add new modes

Bias toward keeping the set small. New modes should answer a question
the existing five don't. Examples:

* `system-cot-suppressor`: system prompt explicitly says "answer
  directly without reasoning". Tests how much natural-language
  instruction can override learned thinking behavior.
* `prefill-skip-thought`: pre-seed the assistant turn with a closed
  thought block (`<|channel>thought\n<channel|>`) to force-skip
  reasoning at the model level rather than relying on server logic.

Both useful but only when investigating a specific failure mode.

## Output layout

```
<snapshot-dir>/
  _run.log              # stdout of the runner
  _summary.json         # rows + run metadata
  _summary.md           # markdown table
  think-default.json    # per-mode {request, response, row}
  nothink.json
  think-low.json
  think-medium.json
  think-raw-noprompt.json
```

The snapshot dir naming follows existing convention:
`<host>-<model-tag>-thinking-control-<YYYY-MM-DD>`.
