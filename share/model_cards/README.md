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
| `complex_problem_max_tokens` | no | For hard reasoning / benchmarking. Used to compute `x-high` and `max` effort tiers. |
| `sampling` | no | Recommended sampler defaults. |
| `reasoning_effort_tiers` | no | Explicit per-tier phase-1 budgets. Overrides any computed defaults. Use this when the ratio-based defaults don't fit the model. |
