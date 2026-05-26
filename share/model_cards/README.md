# Model cards

Sidecar JSON files carrying per-model defaults transcribed from the
upstream model card (typically the HuggingFace README +
`generation_config.json`).

`dflash_server` reads these at startup to set sensible
`--default-max-tokens`, `--think-max-tokens`, sampler, and
`reasoning.effort` tier values for the loaded model. The CLI
flags still override anything here. See
[docs/specs/thinking-budget.md §3](../../docs/specs/thinking-budget.md)
for the resolution order.

## Lookup

The server normalises the loaded GGUF's `general.name` metadata to
lowercase with spaces replaced by `-`, then looks for
`share/model_cards/<normalised>.json`. A missing file falls back to
the per-family table built into the server, then to the hard
fallback (`antirez/ds4 ds4_eval.c` reference values).

## Adding a new card

1. Find the upstream model card (HuggingFace README +
   `generation_config.json`).
2. Note the recommended `max_tokens` (or equivalent), and any
   separate recommendation for hard reasoning / benchmarking
   workloads.
3. Author a JSON file in this directory. Set `source` to the URL
   you used and `verified_at` to today's ISO date.
4. The file is bundled into the Docker image and read at server
   startup. No recompile needed.

## Fields

See [docs/specs/thinking-budget.md §3.3](../../docs/specs/thinking-budget.md)
for the full field reference.

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Display name; informational. |
| `source` | yes | URL of the upstream card. |
| `verified_at` | yes | ISO date these values were last checked. |
| `max_tokens` | yes | The card's standard recommendation. |
| `download_urls` | no | Map of variant tag (e.g. `Q4_K_M`, `bf16`) to GGUF download URL. Used by deployment tooling. |
| `complex_problem_max_tokens` | no | For hard reasoning / benchmarking. Used to compute `x-high` and `max` effort tiers. |
| `sampling` | no | Recommended sampler defaults. |
| `reasoning_effort_tiers` | no | Explicit per-tier phase-1 budgets. Overrides any computed defaults. Use this when the ratio-based defaults don't fit the model. |
| `notes` | no | Free-form notes about provenance, caveats, or non-card-derived choices. |

## Validating a sidecar

The schema for these files lives at [`_schema.json`](_schema.json)
(JSON Schema draft 2020-12). Any author-facing JSON Schema validator
works; a couple of examples:

```bash
# Python (stdlib + jsonschema)
python -m pip install jsonschema
python -c "import json, jsonschema; \
  schema=json.load(open('share/model_cards/_schema.json')); \
  doc=json.load(open('share/model_cards/qwen3.6-27b.json')); \
  jsonschema.Draft202012Validator(schema).validate(doc); print('OK')"

# Node (ajv-cli)
npx --yes ajv-cli@5 validate \
  -s share/model_cards/_schema.json \
  -d share/model_cards/qwen3.6-27b.json \
  --spec=draft2020
```

`additionalProperties: false` is set at the root, so typos in field
names (e.g. `verified_on` instead of `verified_at`) surface as
validation errors instead of being silently ignored by `dflash_server`.
The server itself does a runtime sanity check for the four required
fields when loading a sidecar and warns (does not fail-start) when one
is missing.
