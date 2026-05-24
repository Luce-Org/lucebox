# Model card sidecars

A design spec for the per-model JSON sidecars at `share/model_cards/`.
A sidecar carries values transcribed from the upstream model's
HuggingFace card (or equivalent) so the server can apply
model-appropriate defaults at startup without operator hand-tuning.

This spec covers the file layout, name lookup, JSON schema, field
semantics, resolution order, validation, and authoring workflow.

## 1. Background

GGUF metadata is sparse on inference-time recommendations. A given
GGUF file reliably exposes:

- `general.architecture` — the model family
- `general.name` — a human-readable identifier (varies by source)
- `tokenizer.*` — vocab + special-token ids
- A small handful of sampler hints (e.g. `general.sampling.top_p`)

What it does **not** carry: recommended `max_tokens`, recommended
thinking budget, recommended temperature for benchmark workloads,
or any reasoning-specific knob. Those typically live in the
HuggingFace README + `generation_config.json`, which the GGUF
conversion drops.

To run a model at its trained recommendations without forcing every
operator to memorize them, the server reads a sidecar JSON at
startup. The sidecar is the canonical record of "what does the
upstream card say about this model" and is the primary source of
inference-time defaults.

Sidecars are **source-controlled**, **reviewable**, and **bundled
into the runtime image**. Adding a new model = adding a JSON file.
No recompile required.

## 2. File location and lookup

Sidecars live at:

```
share/model_cards/<name>.json
```

The server resolves `<name>` from the loaded GGUF's `general.name`
metadata, normalized per these rules:

1. Lowercase.
2. Replace spaces with `-`.
3. Strip any character not in `[a-z0-9.-]`.
4. Append `.json`.

Examples:

| GGUF `general.name` | Normalized filename |
|---|---|
| `Qwen3.6-27B` | `qwen3.6-27b.json` |
| `Qwen3.6 27B` | `qwen3.6-27b.json` |
| `Laguna-XS.2` | `laguna-xs.2.json` |
| `DeepSeek-V4-Flash` | `deepseek-v4-flash.json` |

### Cards directory search path

The server probes (in order):

1. `--model-cards-dir <path>` if the operator passed it.
2. `$DFLASH_MODEL_CARDS_DIR` if set.
3. `<binary_dir>/../share/model_cards/` (relative to the running
   binary's parent dir, resolved via `/proc/self/exe`).
4. `share/model_cards/` in the current working directory.

The first directory containing a matching `<name>.json` wins. If
none match, the server falls through to the per-family fallback
table (see §4) and reports the chosen source in the startup banner.

## 3. JSON Schema

Sidecar files are validated against
[`share/model_cards/_schema.json`](../../share/model_cards/_schema.json)
(draft 2020-12). The schema is the authoritative shape; this
section is the human-readable summary.

### 3.1 Required fields

| Field | Type | Purpose |
|---|---|---|
| `name` | string | Display name. Informational — the filename is what matters for lookup. |
| `source` | string (URI) | URL of the upstream card the values were transcribed from. |
| `verified_at` | string (ISO 8601 date, `YYYY-MM-DD`) | When the values were last checked against the source. |
| `max_tokens` | integer ≥ 1 | The card's standard recommended combined cap (reasoning + reply). Drives `default_max_tokens` when no CLI override is set. |

### 3.2 Optional fields

| Field | Type | Purpose |
|---|---|---|
| `download_urls` | object | Map of variant → URL (e.g. `{"Q4_K_M": "https://...", "bf16": "https://..."}`). For operator convenience; the server does not download automatically. |
| `complex_problem_max_tokens` | integer ≥ 1 | The card's recommendation for hard reasoning / benchmark workloads. Drives `x-high` and `max` effort tiers above the standard cap. Omit if the card has no separate complex-problem recommendation. |
| `sampling` | object | Default sampler params. Used to fill values the request body did not specify. Allowed fields: `temperature` (float ≥ 0), `top_p` (float in 0..1), `top_k` (int ≥ 0), `min_p` (float in 0..1), `presence_penalty` (float), `repetition_penalty` (float). |
| `reasoning_effort_tiers` | object | Explicit phase-1 budgets per `reasoning.effort` tier. Keys: `low`, `medium`, `high`, `x-high`, `max` (all optional integers ≥ 0). Missing keys fall through to the computed defaults in `docs/specs/thinking-budget.md` §3.3. |
| `notes` | string | Free-form provenance / caveats. Useful when the card omits values and the sidecar author has to pick defaults from related models or domain knowledge. |

### 3.3 Forbidden

Root-level `additionalProperties` is `false`. Typos like
`reasoning_effort_tier` (singular) are caught at startup with a
warning. This keeps sidecars from accumulating undocumented fields
that look meaningful but aren't.

## 4. Resolution order

For each tunable, the server consults sources in this order. The
first source supplying a value wins:

1. **Explicit CLI flag** (`--think-max-tokens N`,
   `--default-max-tokens N`, `--reasoning-effort-high N`, etc.).
2. **Model card sidecar** (this file).
3. **Per-family fallback table**, built into the C++ server, keyed
   on `general.architecture` (e.g. `qwen35`, `qwen36`, `gemma4`,
   `laguna`). A coarse safety net for known families when no sidecar
   matches.
4. **Hard fallback**, matching `antirez/ds4 ds4_eval.c`'s reference
   values: `default_max_tokens=16000`, `hard_limit_reply_budget=512`,
   `think_max_tokens = default_max_tokens − hard_limit_reply_budget`.

The startup banner prints each tunable's value and which source
supplied it, e.g.:

```
[server] │  max_tokens      = 32768 (share/model_cards/qwen3.6-27b.json)
[server] │  think_max_tokens= 32256 (share/model_cards/qwen3.6-27b.json)
[server] │  effort tiers    = low=4032 (share/model_cards/qwen3.6-27b.json)
[server] │                    medium=16128 (share/model_cards/qwen3.6-27b.json)
[server] │                    ...
```

When a CLI flag overrides a sidecar value, the banner says
`(from CLI)` next to that line. When a value is derived (e.g.
`think_max_tokens` computed from `default_max_tokens` and
`hard_limit_reply_budget`), the banner notes the derivation.

## 5. Field semantics

### 5.1 `max_tokens` and `complex_problem_max_tokens`

`max_tokens` is the card's *standard* recommended combined cap.
The server applies it as the default value for `default_max_tokens`
(used when a request omits `max_tokens`).

`complex_problem_max_tokens`, if present, is the card's
recommendation for hard reasoning / benchmarking workloads. The
server does **not** raise `default_max_tokens` to this value —
operators who want to bench at this scale must pass it explicitly
(e.g. `--max-tokens 81920` on the bench request, or
`--default-max-tokens 81920` on the server). What the field
*does* drive is the upper end of the `reasoning.effort` tier
ladder (see §5.4).

The effective phase-1 cap derived from these fields is
`think_max_tokens = max_tokens − hard_limit_reply_budget`, unless
overridden by `--think-max-tokens`.

### 5.2 `sampling`

Sampler defaults from the card. When a request omits a sampler
field, the server uses the sidecar value. When the request
specifies a field, the request wins (the sidecar is a default, not
a ceiling).

The `sampling` object is forwarded directly into the model's
sampler stack; field names match the OpenAI / Anthropic wire
shape.

There is no CLI override for individual sampling fields today.
Operators who need to force a sampler value should patch the
sidecar (and update `verified_at`) or set it on the request.

### 5.3 `download_urls`

A convenience map. Keys are variant names (`Q4_K_M`, `bf16`, etc.);
values are URLs the operator can `wget`. The server does not
download anything — this is purely documentation. If the model is
gated, the URL still appears in the sidecar; access control
happens at the HF / mirror layer.

### 5.4 `reasoning_effort_tiers`

When present, the sidecar's `reasoning_effort_tiers` object
provides explicit phase-1 budgets per OpenAI-Responses-style
`reasoning.effort` value. Keys are `low`, `medium`, `high`,
`x-high`, `max`. Each value is an integer ≥ 0 representing the
phase-1 cap (in tokens) at that tier.

Tiers are an extension to OpenAI Responses' 3-tier
(`low | medium | high`) vocabulary. The two extra tiers (`x-high`,
`max`) let clients opt in to the card's complex-problem budget when
the prompt warrants it. See `docs/specs/thinking-budget.md` §4.2.

If the field is missing, the server computes defaults from
`max_tokens` + `complex_problem_max_tokens`:

| Tier | Default formula |
|---|---|
| `low` | `think_max × 0.125` |
| `medium` | `think_max × 0.5` |
| `high` | `think_max × 1.0` |
| `x-high` | `(think_max + complex_think_max) / 2` |
| `max` | `complex_think_max` |

Where `think_max = max_tokens − hard_limit_reply_budget` and
`complex_think_max = complex_problem_max_tokens − hard_limit_reply_budget`
(falling back to `think_max` when the card has no complex
recommendation).

The `reasoning_effort_tiers` field exists because the ratio-based
defaults don't fit every model. A smaller model that caps at 8192
tokens has a very different tier curve from Qwen3.6's 32768/81920
envelope, and the model card author can pick more useful values
than a global formula.

Partial overrides are supported: a sidecar can supply only
`low` and `high`, leaving the others to the computed defaults.

## 6. Validation

At startup, after `resolve_model_card` selects a source:

1. **JSON shape**: parsed with `nlohmann::json`. Malformed JSON → log
   error, fall through to family fallback, **do not** fail-start.
2. **Schema check**: each required field present and type-correct.
   Missing required field → log error per field, fall through to
   family fallback.
3. **Field bounds**: `max_tokens ≥ 1`,
   `complex_problem_max_tokens ≥ max_tokens` (if present), tier
   values non-decreasing in `low ≤ medium ≤ high ≤ x-high ≤ max`,
   all values fit under `max_ctx − hard_limit_reply_budget`. Any
   violation → log warning, clamp to satisfy invariants.
4. **Banner**: print the resolved source label, max_tokens,
   think_max_tokens, effort tiers, and sampling defaults.

Validation never fails the server. The design priority is "operator
can always start the server even with a missing/bad sidecar." The
fallback chain (family table → hard fallback) guarantees the server
has working defaults regardless of sidecar quality.

A `validate.sh` script in `share/model_cards/` runs the schema
check at CI time so bad sidecars don't land on `main`.

## 7. Examples

### 7.1 Reasoning-capable, with complex-problem recommendation

[`share/model_cards/qwen3.6-27b.json`](../../share/model_cards/qwen3.6-27b.json):

```json
{
  "name": "Qwen3.6 27B",
  "source": "https://huggingface.co/Qwen/Qwen3.6-27B",
  "verified_at": "2026-05-23",
  "max_tokens": 32768,
  "complex_problem_max_tokens": 81920,
  "sampling": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0
  },
  "reasoning_effort_tiers": {
    "low":    4032,
    "medium": 16128,
    "high":   32256,
    "x-high": 56832,
    "max":    81408
  }
}
```

### 7.2 Non-reasoning, no complex-problem mode

[`share/model_cards/laguna-xs.2.json`](../../share/model_cards/laguna-xs.2.json):

```json
{
  "name": "Laguna-XS.2",
  "source": "https://huggingface.co/Lucebox/Laguna-XS.2-GGUF",
  "verified_at": "2026-05-24",
  "download_urls": {
    "Q4_K_M": "https://huggingface.co/Lucebox/Laguna-XS.2-GGUF/resolve/main/laguna-xs2-Q4_K_M.gguf",
    "bf16":   "https://huggingface.co/Lucebox/Laguna-XS.2-GGUF/resolve/main/laguna-xs2-bf16.gguf"
  },
  "notes": "Non-reasoning MoE code model. Card omits sampler params; defaults below are code-model-typical.",
  "max_tokens": 4096,
  "sampling": {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 50,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0
  }
}
```

Note the absence of `complex_problem_max_tokens` and
`reasoning_effort_tiers` — the server computes degenerate tiers
that all collapse to `think_max` (`3584`), which is fine for a
non-reasoning model since `reasoning.effort` is rarely sent
against it.

## 8. Authoring a new sidecar

1. Find the upstream model card (HuggingFace README +
   `generation_config.json` if available).
2. Note the recommended `max_tokens` (or equivalent). Note any
   separate recommendation for hard reasoning / benchmarking.
3. Author the JSON file in `share/model_cards/`. Set `source` to
   the URL you used and `verified_at` to today's ISO date.
4. Validate against the schema:
   ```bash
   python -m jsonschema -i share/model_cards/<name>.json \
                          share/model_cards/_schema.json
   ```
5. Open a PR. The CI check runs the schema validation against all
   sidecars under `share/model_cards/`.
6. Once merged, the docker image build picks up the new sidecar
   automatically.

When the upstream card changes, bump `verified_at` and review the
fields. A stale `verified_at` (e.g. > 6 months old) is a hint that
the values might be out of date.

## 9. Out of scope

- **Automatic GGUF download.** `download_urls` is operator-facing
  documentation; the server doesn't fetch.
- **Per-effort sampler overrides.** Sampling is a single object
  applied to all effort tiers. There is no way to say "use temp 0.6
  for `effort=high` but 1.0 for `effort=low`."
- **Multi-model deployment.** The server loads one model at a time;
  the sidecar resolution happens once at startup. There is no
  per-request model picker.
- **Card aging / staleness alerts.** `verified_at` is informational
  only; the server doesn't refuse a stale card. A dashboard or CI
  job could surface stale entries — out of scope here.
- **Variants beyond GGUF.** The server only loads GGUF. Non-GGUF
  variants (safetensors, MLX, etc.) get sidecars only if/when those
  loaders ship.
