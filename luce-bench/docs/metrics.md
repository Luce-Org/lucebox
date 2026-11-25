# luce-bench metrics — what each field means

A short reference so the next person scanning a `result.json` doesn't
misread `semantic_hint_rate` as the headline score (as has happened more
than once).

For background on the schema versioning, see
[`src/lucebench/schema.py`](../src/lucebench/schema.py); for the
re-grading flow, see [`src/lucebench/regrade.py`](../src/lucebench/regrade.py).

---

## Headline scores

### `strict_pass_rate`  *(headline)*

The fraction of cases whose extracted answer matches the canonical
answer at the current `grader_version`. This is THE ds4-eval score; in
the markdown report it's the `strict_pass` column.

Encoded as a fraction in [0.0, 1.0] in the canonical JSON; rendered as
percent in the markdown layer. Cross-version comparisons require the
same `grader_version` — the regrade CLI refuses to mix them.

For ds4-eval cases:
* `choice` cases — extractor sweeps for an "Answer: X" marker first, then
  falls back to the last valid letter in the visible text (post-`</think>`).
* `integer` cases — first digit run after the marker, else last digit run.
* `compsec` cases — line-spec subset match: model's lines must be a
  non-empty subset of the expected set (ds4_eval.c semantics).

### `format_pass_rate`

The fraction of cases where the extractor produced a parseable answer
(`given != "?"`). Independent of correctness — a model that confidently
emits `"Answer: Q"` for a choice case with only `A..D` still fails
format. Use it to separate "model never answers in the right shape"
from "model answers wrong".

---

## Diagnostic — NOT the headline

### `semantic_hint_rate`  *(diagnostic, never the score)*

The fraction of cases where the expected answer string appears
**anywhere** in the model's content or reasoning trace. The point is to
flag near-misses: a model that thought of the right number mid-stream
but emitted the wrong format gets a `semantic_hint=True` row even on a
strict-pass failure.

**This is not the score**, and the regrade CLI labels it
`semantic_hint (diag)` to drive that home. Comparing two runs on
`semantic_hint_rate` is meaningless — it's a near-miss heuristic, not
an evaluation metric. The user has been bitten by this enough that the
markdown notes spell it out.

Things `semantic_hint_rate` is NOT:
* not a soft / partial / "semantic" version of `strict_pass_rate`
* not a replacement for a real semantic judge (there isn't one plumbed
  today; see below)

### `semantic_pass_rate`  *(REMOVED — was always 0.0)*

Older `result.json` files emit a `semantic_pass_rate` field that is
ALWAYS `0.0`. There has never been a semantic judge plumbed into the
grader, so the field never had a real meaning — but it sat next to
`strict_pass_rate` in the JSON and got misread repeatedly as "the
semantic score crashed."

`lucebench.normalize.normalize_result` drops this field on load. New
runs do not emit it. **If you wire a real semantic judge in the
future**, emit its score under
`metrics["semantic_judge"][<judge_id>]["pass_rate"]` instead — do NOT
re-introduce a top-level `semantic_pass_rate` field.

---

## Thinking-control verification

### `thinking_control_requested` / `thinking_control_honored`

`thinking_control_requested` is either `"think"` or `"nothink"` — what
the runner asked the server to do. `thinking_control_honored` is a
boolean: did the server actually comply?

When the post-run verify pass hasn't shipped (current state), the
normalizer infers this from rows: in `nothink` mode, any row with
`reasoning_tokens > 0` is a contradicting row and trips the honored
flag to `False`. `contradicting_rows` carries the count so a partial
violation ("the server ignored nothink on 3 of 92 cases") doesn't
silently become "honored=True with caveats."

In the markdown report this surfaces as the `tc_honored` column —
either `honored` or `NO (N)` where N is the contradicting-rows count.

---

## Schema / grader version pinning

### `schema_version` (top-level)

Bumps when the canonical result shape changes
(`lucebench.schema.SCHEMA_VERSION`). Currently `1`.

### `grader_version` (top-level)

A composite of the per-area `GRADER_VERSION` constants — e.g.
`"ds4-eval=1"` for a single-area run, `"ds4-eval=1+gsm8k=1"` if a
future sweep regrade carries multiple areas in one file. Bump the
per-area constant when extractor regexes, line-spec normalisation, or
semantic-hint definitions change.

**Cross-version comparability rule:** the regrade CLI refuses to put
two runs in the same comparison-table row unless their
`grader_version` strings match exactly. To compare runs across a
grader bump, re-grade the older ones (`luce-bench regrade <dir>`)
first.

---

## Pass-rate unit policy

The CANONICAL representation is a fraction in [0.0, 1.0]; the markdown
layer multiplies by 100 for display.

Legacy `result.json` files violate this in two ways: pre-0.2.5 files
encode `pass_rate` as a fraction (0.5761); 0.2.5+ files encode it as a
percent (77.17). The normalizer auto-detects on load using a heuristic
range (≤ 1 → fraction, > 1 ≤ 100 → percent, divide by 100) and tags the
interpretation in `metrics["pass_rate_unit"]` so a downstream consumer
can audit it.

Never read `result["pass_rate"]` directly — go through
`lucebench.normalize.normalize_result`.
