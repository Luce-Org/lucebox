# Model cards

Sidecar JSON files carrying the per-model defaults Lucebox serves with.
They start from the upstream model card (typically the HuggingFace
README + `generation_config.json`); where Lucebox serves differently,
such as greedy sampling, the value is Lucebox's and `notes` keeps the
upstream recommendation.

`luce_server` reads these at startup to set sensible
`--default-max-tokens`, `--think-max-tokens`, sampler, and
`reasoning.effort` tier values for the loaded model, and its launch
profiles: the flags and environment of each qualified hardware setup.
The CLI flags still override anything here. See
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
4. Copy the upstream sampling recipe into `sampling` (and
   `sampling_no_thinking` if the card has an instruct-mode one), and
   set `default_sampling` to `greedy` for a model Lucebox serves.
5. Add the launch profiles Lucebox qualified for the model, and name
   the ones a plain launch should get in `default_profile` (see below).
6. The file is bundled into the Docker image and read at server
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
| `default_sampling` | no | `greedy`: a request that does not sample decodes greedily with no penalties, and `sampling` fills in only for one that sets a temperature above 0. `sampling` (the default): `sampling` fills in for every request. |
| `sampling` | no | The upstream sampler recipe, for fields a request leaves out. `min_p` is read and shown in `/props`, but the sampler does not apply it. |
| `sampling_no_thinking` | no | The same for requests whose thinking is off. |
| `reasoning_effort_tiers` | no | Explicit per-tier phase-1 budgets. Overrides any computed defaults. Use this when the ratio-based defaults don't fit the model. |
| `notes` | no | Free-form notes about provenance, caveats, or non-card-derived choices. |
| `profiles` | no | Launch profiles by name: `aliases`, `summary`, `gpus`, `min_gpu_memory_gib`, `flags`, `env`, `notes` (below). |
| `default_profile` | no | The profile a model served alone gets when its `gpus` come first with its `min_gpu_memory_gib`; with a list, the first that matches. |

## Launch profiles

A profile is a qualified hardware setup for the model: the
`luce_server` flags and the tuning environment it was measured with.

```json
"default_profile": ["ds41-r9700-gorgon", "ds41-gorgon", "ds41-halo"],
"profiles": {
  "ds41-halo": {
    "aliases": ["ds41-lucebox"],
    "summary": "DeepSeek V4.1 Flash on the 128 GB Lucebox: ...",
    "gpus": ["gfx1201", "gfx1151"],
    "flags": {"--target-device": "hip:0", "--expert-device": "hip:1", "--peer-access": true,
              "--max-ctx": "204800"},
    "env": {"LUCE_EXPERT_BUDGET_MB": "10500"}
  },
  "ds41-r9700-gorgon": {
    "summary": "DeepSeek V4.1 Flash on an R9700 beside a 192 GB Gorgon Halo: ...",
    "gpus": ["gfx1201", "gfx1151"],
    "min_gpu_memory_gib": [0, 160],
    ...
  },
  ...
}
```

(abridged from `deepseek-v4.1-flash.json`)

- `aliases`: earlier names of the profile, which `--profile` still
  accepts.
- `gpus`: the GPU architectures the profile was qualified on, from
  `hip:0` upward.
- `min_gpu_memory_gib`: the memory each of `gpus` needs for the
  profile to apply by default, in GiB as KFD reports it (the node's
  memory banks: a discrete GPU's VRAM, the GTT on the Strix Halo), `0`
  for any. It tells apart boards with the same GPUs, such as the
  128 GB Lucebox and an R9700 beside a 192 GB Gorgon Halo.
- `flags`: `luce_server` flags in order, each with its value as a
  string, or `true` for a flag without one. A `share/...` value
  resolves against the install (`server/share` from the repository
  root). `--draft-block-size` applies only to a model that loads its
  drafter in process (`--draft` without `--draft-ipc-bin`), so a
  default profile still serves the model without one.
- `env`: environment defaults, installed at model load.
- `notes`: why the values are what they are.

`--profile <name>` applies a profile by name, from the model's card
or any other; profile names and aliases are unique across the cards.
`--profile none` applies none. Without `--profile`, a model served
alone gets the first of the card's `default_profile` whose `gpus`
start the visible GPUs (KFD order, after `ROCR_VISIBLE_DEVICES` and
`HIP_VISIBLE_DEVICES`), each with its `min_gpu_memory_gib`; on other
hardware nothing applies, and the startup log says why. An explicit
`--profile` applies on any GPUs, with a warning when they are not the
ones it was qualified on. Everything a profile sets is a default: a flag
on the command line replaces the profile's value, `--no-<flag>` leaves
a profile flag out (`--no-peer-access`), and an environment
variable that is already set keeps its value.

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
validation errors instead of being silently ignored by `luce_server`.
The server itself does a runtime sanity check for the four required
fields when loading a sidecar and warns (does not fail-start) when one
is missing.
