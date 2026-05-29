# `/props` endpoint

A design spec for `dflash_server`'s `/props` capability-advertising
endpoint. `/props` is the operator-facing introspection surface: a
single GET that returns enough JSON for a dashboard, a deployment
healthcheck, or a client SDK to know what this server can do and
how it's configured.

This spec covers the URL contract, response shape, per-field
semantics, schema versioning, and backward-compatibility rules.

## 1. Background

There is no industry-standard "what can this LLM server do?"
endpoint. OpenAI exposes `/v1/models` for the model list; Anthropic
exposes nothing equivalent; llama.cpp has `/props` historically as
a server-state snapshot. dflash_server's `/props` extends that
tradition with structured capability advertising for:

- The set of HTTP endpoints exposed
- Reasoning / thinking-budget capability (which effort tiers, what
  budgets)
- Speculative-decode + tool-calling capability
- The loaded model card and its derived settings
- Prefix cache + full cache occupancy
- PFlash (speculative prefill compression) state
- Default sampler params and context length

The intent is "everything an operator needs to verify a deployment
is configured correctly without having to read the startup banner
or grep the process listing."

## 2. Request

```
GET /props HTTP/1.1
```

| Aspect | Detail |
|---|---|
| Method | `GET` |
| Path | `/props` |
| Auth | None. Same posture as `/health` — accessible without bearer token so deployment probes work. |
| Caching | Response is generated per-request and is not cacheable. Fields may change between requests (e.g. `prefix_cache.in_use`, `full_cache.lifetime_hits`). |
| CORS | Allowed by default. Operators can disable with `--no-cors`. |

`/props` never blocks on the worker thread. It reads atomic
counters and config snapshots only. A long-running generate
request will not delay a `/props` response.

## 3. Response

`Content-Type: application/json`. Top-level structure:

```json
{
  "api":                          { "endpoints": [ … ] },
  "budget_envelope":              { … },
  "build":                        { … },
  "build_info":                   "luce-dflash v<ver> props_schema=<n>",
  "capabilities":                 { … },
  "daemon":                       { "alive": true },
  "default_generation_settings":  { … },
  "full_cache":                   { … },
  "host":                         { … } | null,
  "model":                        { "arch": "<string>", "alias": "<string>", "draft_path": "<string|null>", "tokenizer_id": "<string|null>", "target": { … }, "draft": { … } | null },
  "model_alias":                  "<string>",
  "model_card":                   { … } | null,
  "model_path":                   "<string>",
  "pflash":                       { … },
  "prefix_cache":                 { … },
  "reasoning":                    { … },
  "runtime":                      { … },
  "sampling":                     { "capabilities": { … } },
  "server":                       { "name": "<string>", "version": "<string>", "props_schema": <int> },
  "speculative":                  { "enabled": <bool>, "ddtree_budget": <int|null> },
  "speculative_mode":             "off" | "dflash" | "pflash",
  "tool_replay":                  { … }
}
```

All top-level keys are required and always emitted by
`build_props_body`. Optional nested fields may be `null` when the
corresponding feature is disabled (e.g. `pflash.threshold` when
`pflash.enabled = false`; `speculative.ddtree_budget` when
`speculative.enabled = false`). `model_card` itself is `null` when
the server fell through to family or hard fallback (no sidecar
matched).

## 4. Per-section field semantics

### 4.1 `api`

```json
"api": {
  "endpoints": [
    "GET /health",
    "GET /props",
    "GET /v1/models",
    "POST /v1/chat/completions",
    "POST /v1/messages",
    "POST /v1/messages/count_tokens",
    "POST /v1/responses"
  ]
}
```

`endpoints` is an unordered list of method-and-path strings the
server actually handles. Extension points (e.g. tool-call
endpoints, embedding endpoints) appear here when implemented.

### 4.2 `budget_envelope`

```json
"budget_envelope": {
  "model_card_source":       "share/model_cards/qwen3.6-27b.json",
  "default_max_tokens":      32768,
  "hard_limit_reply_budget": 512,
  "think_max_tokens":        32256,
  "effort_tiers": {
    "low":    4032,
    "medium": 16128,
    "high":   32256,
    "x-high": 56832,
    "max":    81408
  }
}
```

The runtime-resolved budget knobs driving the thinking-budget
envelope. These are always-present, even when no sidecar was loaded
(see §4.10 `model_card`). They may differ from the authored card
values because of CLI overrides (`--default-max-tokens`,
`--think-max-tokens`, `--reasoning-effort-<tier>`) and the
absolute-tier ceiling clamping (spec §3.5).

- `model_card_source` — string, always present. The lookup hit that
  produced these values: one of `share/model_cards/<file>.json`,
  `family:<arch>`, or `hard-fallback`. Matches the startup banner.
- `default_max_tokens` — effective combined cap (reasoning + reply)
  applied when a request omits `max_tokens`. May diverge from
  `model_card.max_tokens` if the operator passed `--max-tokens` or
  `--default-max-tokens`.
- `hard_limit_reply_budget` — effective reply-reserve ceiling in
  tokens.
- `think_max_tokens` — effective phase-1 (reasoning) ceiling.
  Derived as `default_max_tokens − hard_limit_reply_budget` unless
  `--think-max-tokens` overrides.
- `effort_tiers` — phase-1 budgets per `reasoning.effort` tier.
  Resolved from the card's `reasoning_effort_tiers` (if present),
  then computed from `max_tokens` / `complex_problem_max_tokens` per
  spec §3.3, then clamped to `max_ctx − hard_limit_reply_budget`
  per §3.5. May differ from `model_card.reasoning_effort_tiers`
  because of that clamp.

`budget_envelope` is the source of truth for what the server will
actually do with a request; `model_card` (§4.10) is the source of
truth for what the authored card says.

### 4.3 `build_info` (legacy) and `build` (schema 3+)

```
"build_info": "luce-dflash v0.0.0+cpp props_schema=4"
"build": {
  "server_name":    "luce-dflash",
  "server_version": "0.0.0+cpp",
  "props_schema":   4,
  "git_sha":        "6d12378…",
  "image_tag":      "sha-6d12378-cuda12",
  "image_digest":   null,
  "build_time":     "2026-05-28T13:43:57Z"
}
```

`build_info` is the legacy single-string identity (server name,
build version, `props_schema`). Schema version bumps when the
response shape changes (see §5). Retained verbatim for back-compat
— consumers that grep `build_info` keep working without changes.

`build` (schema 3+) is the structured replacement and the
recommended source of truth for "what binary is running":

- `server_name` / `server_version` / `props_schema` mirror the
  identity fields. Repeated here so a single `curl … | jq .build`
  returns everything an operator needs.
- `git_sha` — full git commit sha of the source tree the image
  was built from. Set by CI from `${{ github.sha }}` via
  `docker-bake.hcl`; set locally by `scripts/build_image.sh` from
  `git rev-parse HEAD`. `null` on bare-metal builds (no
  `/opt/lucebox-hub/IMAGE_INFO` file).
- `image_tag` — headline tag the image was published under
  (e.g. `cuda12`, `sha-6d12378-cuda12`, `0.3.0-cuda12`). Set by
  CI from `docker/metadata-action`'s `version` output. `null`
  outside Docker.
- `image_digest` — reserved for future use. The
  content-addressable registry digest would let an operator pin
  `ghcr.io/.../lucebox-hub@sha256:…` after a pull; we don't query
  the Docker socket from inside the container today, so this is
  always `null`. Kept in the schema so adding it later is
  additive.
- `build_time` — ISO 8601 UTC timestamp the image was built at.
  `null` outside Docker.

The `build.image_*` fields are populated from
`/opt/lucebox-hub/IMAGE_INFO`, which `Dockerfile` writes from the
`GIT_SHA`, `IMAGE_TAG`, and `BUILD_TIME` build args. The path can
be overridden with `$DFLASH_IMAGE_INFO_PATH` (used by unit tests
to inject fixtures).

### 4.4 `capabilities`

```json
"capabilities": {
  "reasoning_supported":   true,
  "speculative_supported": true,
  "tools_supported":       true
}
```

Coarse-grained boolean feature flags. Each corresponds to a major
client capability:

- `reasoning_supported` — server accepts `thinking:{type:"enabled"}`
  and `reasoning:{effort:...}` request fields. If `false`, those
  fields are silently ignored.
- `speculative_supported` — DDTree speculative decode is wired up
  for the loaded backend. If `false`, requests run pure AR.
- `tools_supported` — server accepts `tools` and `tool_choice`
  fields and emits `tool_calls` blocks. If `false`, those fields
  are ignored.

### 4.5 `daemon`

```json
"daemon": { "alive": true }
```

`alive` is `true` if the model backend is loaded and ready. A
`false` value indicates the server is up but the daemon thread
crashed or is restarting. Healthchecks should treat `false` as a
failure even if HTTP returns 200.

### 4.6 `default_generation_settings`

```json
"default_generation_settings": {
  "min_p":          0.0,
  "n_ctx":          98304,
  "repeat_penalty": 1.0,
  "temperature":    0.0,
  "top_k":          0,
  "top_p":          1.0
}
```

The default sampler values the server applies when a request omits
the corresponding field. `n_ctx` is the maximum prompt+output
context length (= `--max-ctx`).

Field names use llama.cpp conventions (`repeat_penalty`, not
`repetition_penalty`) for compatibility with `/props` consumers
written against llama-server.

These values are the server's hard-coded sampler defaults
(`temperature=0.0`, `top_p=1.0`, `top_k=0`, `min_p=0.0`,
`repeat_penalty=1.0`) and **do not** reflect the loaded model
card's `sampling` block. The model card's sampler defaults are
applied at request-parse time when a request omits a sampler
field; `/props` only carries the server-wide knobs. To see the
card's sampler values, read the sidecar JSON or the startup
banner.

### 4.7 `full_cache`

```json
"full_cache": {
  "capacity":      0,
  "disk_bytes":    0,
  "enabled":       false,
  "in_use":        0,
  "lifetime_hits": 0
}
```

The full-prompt cache (disk-backed). When `enabled = false`,
`capacity`, `in_use`, `lifetime_hits`, and `disk_bytes` are all
zero and ignored.

Counters use atomic loads (`std::memory_order_relaxed`); the
snapshot is tear-free per field but the set of fields is **not
internally consistent** — e.g. `in_use` and `lifetime_hits` may
correspond to slightly different points in wall time. Acceptable
for an introspection report; not safe for control-flow decisions.

### 4.8 `model`

```json
"model": {
  "arch":         "qwen35",
  "alias":        "dflash",
  "draft_path":   "/path/to/draft.gguf" | null,
  "tokenizer_id": "qwen3" | null,
  "target": {
    "path":       "/path/to/Qwen3.6-27B-Q4_K_M.gguf",
    "size_bytes": 17134510080,
    "sha256":     "abc123…",
    "gguf": {
      "general.architecture":         "qwen35",
      "general.name":                 "Qwen3.6-27B",
      "general.file_type":            15,
      "general.file_type_name":       "Q4_K_M",
      "general.quantization_version": 2,
      "block_count":                  64,
      "embedding_length":             5120,
      "context_length":               65536,
      "vocab_size":                   152064
    }
  },
  "draft": { … } | null
}
```

`arch` is the `general.architecture` value from the loaded GGUF,
normalized. `tokenizer_id` is a best-effort tokenizer family hint
from GGUF metadata.

`alias` (schema 3+) mirrors the top-level `model_alias` for
grouping under `model` alongside the rest of the model identity.
The top-level `model_alias` stays for back-compat.

`draft_path` (schema 1+, legacy) is the speculative-decode draft
GGUF path, or `null` when no draft is loaded. New consumers should
prefer `model.draft.path` — same value, but grouped with the rest
of the draft identity.

`target` (schema 3+) is the full identity of the loaded target
weights. Always present and non-null when the server is up — a
`null` `target` indicates a load failure that should have aborted
boot, so it's a strong signal something is wrong.

`draft` (schema 3+) is the same identity payload for the draft
GGUF, or **explicit JSON null** when `--draft` was not passed.
The normal target-only configurations are `laguna` and the
`qwen3.6-moe` preset; explicit-null (not omitted) lets consumers
distinguish "no draft loaded" from "field not in this schema
version."

#### `model.target` / `model.draft` field shape

| field        | type                  | meaning |
|---|---|---|
| `path`       | `string`              | Absolute filesystem path of the loaded GGUF. |
| `size_bytes` | `integer \| null`     | File size from `stat()`. `null` if the stat failed. |
| `sha256`     | `string \| null`      | Lowercase hex sha256 (64 chars). Cached to a `<path>.sha256` sidecar so subsequent restarts skip the rehash. `null` when `$DFLASH_SKIP_SHA256=1` or the file couldn't be opened for reading. |
| `gguf`       | `object`              | Selected `general.*` and `<arch>.*` header fields. Each field is `null` when the GGUF doesn't carry the corresponding key — drafter GGUFs in particular omit `context_length` and `vocab_size` more often than full target models do. |

The `gguf` sub-object's keys map 1:1 to GGUF metadata keys:

- `general.architecture` — raw architecture string (e.g. `qwen35`,
  `qwen3`, `gemma4`, `laguna`).
- `general.name` — display name from the GGUF.
- `general.file_type` — raw `LLAMA_FTYPE_*` integer (see
  `server/deps/llama.cpp/include/llama.h` for the full table).
  15 = Q4_K_M, 17 = Q5_K_M, 30 = IQ4_XS, 32 = BF16, etc.
- `general.file_type_name` — operator-friendly decoded tag for
  `general.file_type` (e.g. `Q4_K_M`, `IQ4_XS`, `BF16`).
- `general.quantization_version` — bumped on quant-format changes
  (2 is the current value for K-quants and IQ-quants).
- `block_count` — `<arch>.block_count` (number of transformer
  blocks).
- `embedding_length` — `<arch>.embedding_length` (model hidden
  size).
- `context_length` — `<arch>.context_length` (max context the
  weights themselves were trained for; may exceed the server's
  runtime `n_ctx` cap).
- `vocab_size` — `<arch>.vocab_size` or the length of
  `tokenizer.ggml.tokens` (fallback when the key isn't written).

The sha256 is computed once at startup. For a multi-GB target
GGUF this is ~30s on a fast NVMe; the result is written to a
sidecar file `<path>.sha256` so subsequent restarts read it from
disk instead of rehashing. Set `$DFLASH_SKIP_SHA256=1` to disable
hashing entirely (faster cold start, but `sha256` will be `null`
at /props).

### 4.9 `model_alias` and `model_path`

`model_alias` is the value clients should pass as the `model` field
in chat/responses requests (defaults to `"dflash"`; override with
`--model-name`).

`model_path` is the absolute filesystem path of the loaded target
GGUF. Useful for "which weights is this server actually serving"
checks.

### 4.10 `model_card`

```json
"model_card": {
  "name":                       "Qwen3.6 27B",
  "source":                     "https://huggingface.co/Qwen/Qwen3.6-27B",
  "verified_at":                "2026-05-23",
  "max_tokens":                 32768,
  "complex_problem_max_tokens": 81920,
  "sampling": {
    "temperature":        1.0,
    "top_p":              0.95,
    "top_k":              20,
    "min_p":              0.0,
    "presence_penalty":   0.0,
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

The on-disk model-card sidecar that was loaded for this run,
emitted **verbatim** (1:1 with the JSON in
`share/model_cards/<name>.json`). The shape validates against
[`share/model_cards/_schema.json`](../../share/model_cards/_schema.json) —
all of `name`, `source`, `verified_at`, `max_tokens`,
`complex_problem_max_tokens`, `sampling`, `reasoning_effort_tiers`,
`download_urls`, and `notes` appear here exactly as the sidecar
author wrote them.

`model_card` is `null` when the server fell through to a per-family
fallback or the hard fallback (no `share/model_cards/<stem>.json`
matched the loaded GGUF's `general.name`). In that case the resolved
budget knobs still appear under `budget_envelope` (§4.2), tagged
with `model_card_source` = `"family:<arch>"` or `"hard-fallback"`.

The `source` field inside `model_card` is the **upstream model-card
URL** (e.g. the HuggingFace card the sidecar was transcribed from),
NOT the filepath the server loaded. The filepath / lookup-hit label
lives at `budget_envelope.model_card_source` (§4.2). This split
keeps the on-disk sidecar contract pure (authored JSON, schema-validated)
and the runtime-resolution metadata in its own section.

For the runtime-resolved budget values (`default_max_tokens`,
`think_max_tokens`, effort tiers post-clamp) the server will actually
apply, see `budget_envelope` (§4.2).

See [`docs/specs/model-cards.md`](model-cards.md) for the sidecar
format and resolution order.

### 4.11 `pflash`

```json
"pflash": {
  "bsa_alpha":   null,
  "bsa_enabled": null,
  "drafter_gguf": null,
  "enabled":     false,
  "keep_ratio":  null,
  "lm_head_fix": null,
  "mode":        "off",
  "skip_park":   null,
  "threshold":   null
}
```

PFlash (speculative prefill compression) state. When `enabled =
false` and/or `mode = "off"`, all other fields are `null`. When
enabled, fields carry the runtime configuration:

- `mode` — `"off" | "auto" | "always"`
- `threshold` — token-count threshold for AUTO mode
- `keep_ratio` — fraction of tokens retained after compression
- `drafter_gguf` — path to the compression drafter GGUF
- `skip_park` — whether to skip park/unpark (large-VRAM GPUs)
- `bsa_enabled` / `bsa_alpha` / `lm_head_fix` — backend-specific
  PFlash tunables

### 4.12 `prefix_cache`

```json
"prefix_cache": {
  "capacity":      0,
  "in_use":        0,
  "lifetime_hits": 0
}
```

The inline prefix cache (system-prompt KV reuse). Same atomic /
non-strictly-consistent semantics as `full_cache` (§4.7).
`capacity = 0` means the cache is disabled.

### 4.13 `reasoning`

```json
"reasoning": {
  "default":         null | "low" | "medium" | "high" | "x-high" | "max",
  "supported":       true,
  "supported_efforts": ["low", "medium", "high", "x-high", "max"]
}
```

Reasoning capability:

- `supported` — does the server accept `reasoning.effort` and
  `thinking:{type:"enabled"}` request fields. When `false`, those
  fields are silently ignored (and the rest of this section can be
  ignored).
- `supported_efforts` — the full set of effort tier values the
  server will recognize. dflash_server always lists all five
  (`low`, `medium`, `high`, `x-high`, `max`); other servers may
  list a subset.
- `default` — when set, the effort tier the server will apply if
  a request enables thinking without specifying `effort`. `null`
  means no default (request must specify).

See [`docs/specs/thinking-budget.md`](thinking-budget.md) §4 for
the per-tier semantics.

### 4.14 `sampling.capabilities`

```json
"sampling": {
  "capabilities": {
    "supports_frequency_penalty": true,
    "supports_seed":              true,
    "supports_temperature":       true,
    "supports_top_k":             true,
    "supports_top_p":             true
  }
}
```

Which sampler fields the server honors on requests. A `false`
value means the field is silently ignored in request bodies.
Clients can use this to skip sending unsupported fields and avoid
confusion when behavior doesn't match the request.

The `sampling` object intentionally nests `capabilities` for
future expansion (e.g. `sampling.defaults`, though that lives in
§4.6 today).

### 4.15 `speculative`

```json
"speculative": {
  "ddtree_budget": 22
}
```

Speculative-decode runtime state. When `capabilities.speculative_supported = false`,
this section is omitted entirely.

- `ddtree_budget` — current DDTree budget (= `--ddtree-budget`).

Future fields: `accept_rate`, `lookahead_depth`, `draft_model_id`
— added as the speculative-decode surface grows.

### 4.16 `runtime`

```json
"runtime": {
  "backend":         "cuda",
  "fa_window":       2048,
  "kv_cache_k":      "q4_0",
  "kv_cache_v":      "q4_0",
  "lazy_draft":      false,
  "target_sharding": false,
  "chunk":           512,
  "target_device":   "auto:0",
  "draft_device":    "auto:0"
}
```

Runtime knobs resolved at startup. These reflect the effective
configuration the server is running with — CLI overrides, model-
card-driven defaults, and binary fallback defaults are all
collapsed into one snapshot. Bench/snapshot tooling reads this
wholesale into `result.json.server_info` so post-hoc forensics on
configuration drift between runs is possible.

- `backend` — active compute backend: `"cuda" | "hip" | "cpu"`.
- `fa_window` — sliding-window attention window in tokens.
- `kv_cache_k` / `kv_cache_v` — effective KV cache dtypes (e.g.
  `"q4_0"`, `"tq3_0"`, `"f16"`). Operator's CLI choice when set,
  otherwise the binary's auto-default (`tq3_0` when
  `max_ctx > 6144`, else `q4_0`, on CUDA).
- `lazy_draft` — whether the decode draft is parked when idle.
- `target_sharding` — true when the target model is layer-split
  across multiple GPUs.
- `chunk` — prefill chunk size in tokens. Determines how prompt
  tokens are batched into the target model during prefill.
- `target_device` — resolved target-model device placement string
  (e.g. `"auto:0"`, `"cuda:0"`).
- `draft_device` — resolved draft-model device placement, or
  `null` when no draft model is loaded.

### 4.17 `host` (schema 4+)

```json
"host": {
  "os_pretty":         "Ubuntu 22.04.3 LTS",
  "kernel":            "6.6.87.2-microsoft-standard-WSL2",
  "wsl_version":       "wsl2",
  "docker_version":    "29.1.3",
  "nvidia_driver":     "596.36",
  "nvidia_ctk_version":"1.16.2",
  "cpu_model":         "Intel(R) Core(TM) Ultra 9 275HX",
  "nproc":             24,
  "ram_gb":            64,
  "gpus": [
    {
      "index":         0,
      "uuid":          "GPU-abc…",
      "pci_bus_id":    "00000000:01:00.0",
      "name":          "NVIDIA GeForce RTX 5090 Laptop GPU",
      "sm":            "12.0",
      "vram_gb":       24,
      "power_limit_w": 175
    }
  ],
  "cuda_visible_devices": "0",
  "source":            "lucebox.sh",
  "collector":         "lucebox.sh",
  "collected_at":      "2026-05-28T20:31:42Z"
}
```

Host-identity facts captured at container startup by
`server/scripts/entrypoint.sh` from the `LUCEBOX_HOST_*` env vars the
host wrapper (`lucebox.sh::probe_host`) exports. Written to
`/opt/lucebox-hub/HOST_INFO` (path override: `$DFLASH_HOST_INFO_PATH`
for tests) and read verbatim into `ServerConfig.host_info` by
`server_main::read_host_info`.

Surfaces so every benchmark snapshot can self-classify the rig it
ran on, even when the snapshot dir is pulled out of context months
later. `luce-bench snapshot` writes this into `host.json` and into
each per-area `<area>.json` so individual area files self-describe.

`null` when `/opt/lucebox-hub/HOST_INFO` is missing or malformed —
the normal case for bare-metal dev builds that bypass the container
entrypoint entirely. Containers launched by `lucebox.sh` always get a
populated block; containers launched directly via `docker run` get a
stub `{"source": "unknown", "collector": "entrypoint.sh", ...}` so
the block is always present in container deployments.

Fields:

- `os_pretty` — string|null. `PRETTY_NAME` from
  `/etc/os-release`. e.g. `"Ubuntu 22.04.3 LTS"`.
- `kernel` — string|null. `uname -r` on the host.
- `wsl_version` — `"wsl1"`, `"wsl2"`, or `null`. `"wsl2"` matches
  the modern `microsoft-standard-WSL2` kernel string; `"wsl1"` is
  the legacy translation layer; `null` is bare Linux / macOS.
- `docker_version` — string|null. Docker server version from
  `docker version --format '{{.Server.Version}}'`.
- `nvidia_driver` — string|null. Driver version from `nvidia-smi`.
- `nvidia_ctk_version` — string|null. NVIDIA Container Toolkit
  version (`nvidia-ctk --version`). Distinct from `docker_version` —
  the runtime that wires GPUs into containers can lag behind the
  daemon.
- `cpu_model` — string|null. First `"model name"` from
  `/proc/cpuinfo`.
- `nproc` — int|null. Logical CPU count.
- `ram_gb` — int|null. Total RAM in GB.
- `gpus` — array of objects (possibly empty). One entry per
  installed GPU; the array preserves nvidia-smi's enumeration
  order. Per-entry fields: `index` (int), `uuid` (string),
  `pci_bus_id` (string), `name` (string), `sm` (string,
  compute capability like `"12.0"`), `vram_gb` (int),
  `power_limit_w` (int, may differ from manufacturer spec when
  the operator has set a power cap).
- `cuda_visible_devices` — string|null. Mirrors the env var; `null`
  means "all GPUs visible".
- `source` — string. One of `"lucebox.sh"`, `"unknown"`. Indicates
  how the block was populated.
- `collector` — string. The script that wrote HOST_INFO: usually
  `"lucebox.sh"` when the host wrapper drove the run, or
  `"entrypoint.sh"` on the stub-only path.
- `collected_at` — ISO 8601 UTC timestamp string.

## 5. Schema versioning

`build_info` includes `props_schema=<n>`, mirrored in
`server.props_schema` and (schema 3+) `build.props_schema`. The
integer `n` bumps when fields are added or changed; consumers
should treat unknown fields as ignorable. The current schema is
`4`.

### 5.0 Changelog

- **`4`** — Additive over `3`. New top-level `host` object — host-
  identity facts (OS, kernel, WSL version, docker version, NVIDIA
  driver, NVIDIA Container Toolkit version, CPU model, nproc, RAM,
  per-GPU array with UUID/PCI/SM/VRAM/power, CUDA_VISIBLE_DEVICES)
  captured by `server/scripts/entrypoint.sh` from the
  `LUCEBOX_HOST_*` env the host wrapper exports. `null` when
  `/opt/lucebox-hub/HOST_INFO` is missing (bare-metal dev). Pre-4
  consumers ignore the new key; new consumers (luce-bench's
  snapshot subcommand in particular) gate on the version to know
  the block is guaranteed-present, and fall back to a client-side
  hostinfo probe against pre-4 servers.
- **`3`** — Additive over `2`. New top-level `build` object — a
  structured replacement for the single-string `build_info` that
  carries `git_sha`, `image_tag`, and `build_time` baked into the
  container at build time. New `model.target` and `model.draft`
  sub-objects carry full GGUF identity (path, `size_bytes`,
  `sha256`, and `gguf.*` header fields including
  `general.file_type[_name]`, `block_count`, `embedding_length`,
  `context_length`, `vocab_size`). New `model.alias` field
  (mirror of top-level `model_alias`). The pre-3 top-level
  `build_info`, `model_path`, `model_alias`, and
  `model.draft_path` stay verbatim for back-compat. Schema is
  still bumped (vs leaving at `2`) so consumers can negotiate
  the new fields and lucebench's preflight can switch its
  display format based on the version.
- **`2`** — `model_card` is now the wholesale on-disk sidecar JSON
  (or `null` when family/hard fallback was used). Runtime-resolved
  budget knobs that used to live under `model_card`
  (`hard_limit_reply_budget`, `think_max_tokens`, `effort_tiers`,
  effective `max_tokens`) moved to a new top-level `budget_envelope`
  section. The `source` field inside `model_card` is now the
  upstream URL from the sidecar; the lookup-hit filepath / label
  lives at `budget_envelope.model_card_source`.
- **`1`** — Initial schema.

### 5.1 Non-breaking changes

Pure additive changes — new top-level section, new field inside
an existing section, new entry in `api.endpoints` or
`reasoning.supported_efforts`, loosened field bounds — historically
did not bump `props_schema`. Schema 3 is a deliberate exception:
it's additive (new `build`, `model.target`, `model.draft`,
`model.alias`) but bumps the version so consumers (lucebench's
preflight in particular) can opt in to the new display when the
fields are guaranteed-present, and fall back when talking to an
older server. The rule going forward: **bumps are allowed for
additive changes too** — pre-3 clients keep working because they
ignore unknown fields; new clients gate on the version to know
they can rely on the new shape.

### 5.2 Breaking changes (bump `props_schema`)

- Removing a field or section.
- Renaming a field.
- Changing a field's type.
- Tightening field bounds in a way that may invalidate previously
  valid values.

After a bump, the server may continue to emit the old shape under a
compat flag for one minor version; consult the changelog when the
version increments.

## 6. Example: full response

```json
{
  "api": {
    "endpoints": [
      "GET /health",
      "GET /props",
      "GET /v1/models",
      "POST /v1/chat/completions",
      "POST /v1/messages",
      "POST /v1/messages/count_tokens",
      "POST /v1/responses"
    ]
  },
  "budget_envelope": {
    "model_card_source":       "share/model_cards/qwen3.6-27b.json",
    "default_max_tokens":      32768,
    "hard_limit_reply_budget": 512,
    "think_max_tokens":        32256,
    "effort_tiers": {
      "low":    4032,
      "medium": 16128,
      "high":   32256,
      "x-high": 56832,
      "max":    81408
    }
  },
  "build": {
    "server_name":    "luce-dflash",
    "server_version": "0.0.0+cpp",
    "props_schema":   4,
    "git_sha":        "6d12378abc456789012345678901234567890abcd",
    "image_tag":      "sha-6d12378-cuda12",
    "image_digest":   null,
    "build_time":     "2026-05-28T13:43:57Z"
  },
  "build_info": "luce-dflash v0.0.0+cpp props_schema=4",
  "host": {
    "os_pretty":          "Ubuntu 22.04.3 LTS",
    "kernel":             "6.6.87.2-microsoft-standard-WSL2",
    "wsl_version":        "wsl2",
    "docker_version":     "29.1.3",
    "nvidia_driver":      "596.36",
    "nvidia_ctk_version": "1.16.2",
    "cpu_model":          "Intel(R) Core(TM) Ultra 9 275HX",
    "nproc":              24,
    "ram_gb":             64,
    "gpus": [
      {
        "index":          0,
        "uuid":           "GPU-abc",
        "pci_bus_id":     "00000000:01:00.0",
        "name":           "NVIDIA GeForce RTX 5090 Laptop GPU",
        "sm":             "12.0",
        "vram_gb":        24,
        "power_limit_w":  175
      }
    ],
    "cuda_visible_devices": "0",
    "source":             "lucebox.sh",
    "collector":          "lucebox.sh",
    "collected_at":       "2026-05-28T20:31:42Z"
  },
  "capabilities": {
    "reasoning_supported":   true,
    "speculative_supported": true,
    "tools_supported":       true
  },
  "daemon": { "alive": true },
  "default_generation_settings": {
    "min_p":          0.0,
    "n_ctx":          98304,
    "repeat_penalty": 1.0,
    "temperature":    1.0,
    "top_k":          20,
    "top_p":          0.95
  },
  "full_cache": {
    "capacity":      0,
    "disk_bytes":    0,
    "enabled":       false,
    "in_use":        0,
    "lifetime_hits": 0
  },
  "model": {
    "arch":         "qwen35",
    "alias":        "dflash",
    "draft_path":   "/.../dflash-draft-3.6-q8_0.gguf",
    "tokenizer_id": "qwen3",
    "target": {
      "path":       "/.../Qwen3.6-27B-Q4_K_M.gguf",
      "size_bytes": 17134510080,
      "sha256":     "abc123def456789012345678901234567890abcd0123456789abcdef01234567",
      "gguf": {
        "general.architecture":         "qwen35",
        "general.name":                 "Qwen3.6-27B",
        "general.file_type":            15,
        "general.file_type_name":       "Q4_K_M",
        "general.quantization_version": 2,
        "block_count":                  64,
        "embedding_length":             5120,
        "context_length":               65536,
        "vocab_size":                   152064
      }
    },
    "draft": {
      "path":       "/.../dflash-draft-3.6-q8_0.gguf",
      "size_bytes": 425000000,
      "sha256":     "deadbeef00112233445566778899aabbccddeeff00112233445566778899aabb",
      "gguf": {
        "general.architecture":         "qwen3",
        "general.name":                 "Qwen3-0.6B-DFlash-draft",
        "general.file_type":            7,
        "general.file_type_name":       "Q8_0",
        "general.quantization_version": 2,
        "block_count":                  28,
        "embedding_length":             1024,
        "context_length":               32768,
        "vocab_size":                   152064
      }
    }
  },
  "model_alias": "dflash",
  "model_card": {
    "name":                       "Qwen3.6 27B",
    "source":                     "https://huggingface.co/Qwen/Qwen3.6-27B",
    "verified_at":                "2026-05-23",
    "max_tokens":                 32768,
    "complex_problem_max_tokens": 81920,
    "sampling": {
      "temperature":        1.0,
      "top_p":              0.95,
      "top_k":              20,
      "min_p":              0.0,
      "presence_penalty":   0.0,
      "repetition_penalty": 1.0
    },
    "reasoning_effort_tiers": {
      "low":    4032,
      "medium": 16128,
      "high":   32256,
      "x-high": 56832,
      "max":    81408
    }
  },
  "model_path": "/.../Qwen3.6-27B-Q4_K_M.gguf",
  "pflash": {
    "bsa_alpha":   null,
    "bsa_enabled": null,
    "drafter_gguf": null,
    "enabled":     false,
    "keep_ratio":  null,
    "lm_head_fix": null,
    "mode":        "off",
    "skip_park":   null,
    "threshold":   null
  },
  "prefix_cache": {
    "capacity":      0,
    "in_use":        0,
    "lifetime_hits": 0
  },
  "reasoning": {
    "default":           null,
    "supported":         true,
    "supported_efforts": ["low", "medium", "high", "x-high", "max"]
  },
  "sampling": {
    "capabilities": {
      "supports_frequency_penalty": true,
      "supports_seed":              true,
      "supports_temperature":       true,
      "supports_top_k":             true,
      "supports_top_p":             true
    }
  },
  "speculative": {
    "ddtree_budget": 22
  }
}
```

## 7. Out of scope

- **Authentication.** Same posture as `/health`: open. A future
  version may add an opt-in auth requirement, but that's a separate
  spec.
- **Per-client capability negotiation.** `/props` is a static
  advertise; there is no `GET /props?for_client=X` form.
- **Server-pushed updates.** `/props` is request-response. There is
  no streaming variant.
- **Runtime metrics** (tokens/sec, accept rate, queue depth).
  Those belong in a dedicated `/metrics` endpoint (Prometheus
  format), not in `/props`. The hits/usage counters here are
  *configuration* state, not *performance* metrics.
- **Multi-model.** The server loads one model; `/props` describes
  it. A multi-model server is out of scope for this design.
