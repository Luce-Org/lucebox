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
  "build_info":                   "luce-dflash v<ver> props_schema=<n>",
  "capabilities":                 { … },
  "daemon":                       { "alive": true },
  "default_generation_settings":  { … },
  "full_cache":                   { … },
  "model":                        { … },
  "model_alias":                  "<string>",
  "model_card":                   { … },
  "model_path":                   "<string>",
  "pflash":                       { … },
  "prefix_cache":                 { … },
  "reasoning":                    { … },
  "sampling":                     { "capabilities": { … } },
  "speculative":                  { … }
}
```

All top-level keys are required. Optional nested fields may be
`null` when the corresponding feature is disabled (e.g.
`pflash.threshold` when `pflash.enabled = false`).

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

### 4.2 `build_info`

```
"build_info": "luce-dflash v0.0.0+cpp props_schema=1"
```

A single string carrying: server name, build version, and the
**`props_schema` version**. Schema version bumps when the response
shape changes in a non-backward-compatible way (see §5).

### 4.3 `capabilities`

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

### 4.4 `daemon`

```json
"daemon": { "alive": true }
```

`alive` is `true` if the model backend is loaded and ready. A
`false` value indicates the server is up but the daemon thread
crashed or is restarting. Healthchecks should treat `false` as a
failure even if HTTP returns 200.

### 4.5 `default_generation_settings`

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

When the loaded model card carries a `sampling` object, those
values replace the C++ defaults in this section.

### 4.6 `full_cache`

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

### 4.7 `model`

```json
"model": {
  "arch":         "qwen35",
  "draft_path":   "/path/to/draft.gguf" | null,
  "tokenizer_id": "qwen3" | null
}
```

`arch` is the `general.architecture` value from the loaded GGUF,
normalized. `draft_path` is the speculative-decode draft model
path, or `null` when no draft is loaded. `tokenizer_id` is a
best-effort tokenizer family hint from GGUF metadata.

### 4.8 `model_alias` and `model_path`

`model_alias` is the value clients should pass as the `model` field
in chat/responses requests (defaults to `"dflash"`; override with
`--model-name`).

`model_path` is the absolute filesystem path of the loaded target
GGUF. Useful for "which weights is this server actually serving"
checks.

### 4.9 `model_card`

```json
"model_card": {
  "source":                  "share/model_cards/qwen3.6-27b.json",
  "max_tokens":              32768,
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

The resolved model card values driving the thinking-budget envelope.

- `source` — the source label of the resolved card (sidecar path,
  `family:<arch>`, or `hard-fallback`). Matches what the startup
  banner printed.
- `max_tokens` — effective `default_max_tokens`.
- `hard_limit_reply_budget` — effective reply-reserve ceiling.
- `think_max_tokens` — effective phase-1 ceiling.
- `effort_tiers` — phase-1 budgets per `reasoning.effort` tier.

When `--think-max-tokens` is overridden on the CLI, `think_max_tokens`
here reflects the CLI value (not the card's recommendation). The
banner is the place to see "from CLI" vs "from sidecar" provenance;
`/props` only carries the final effective values.

See [`docs/specs/model-cards.md`](model-cards.md) for the sidecar
format and resolution order.

### 4.10 `pflash`

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

### 4.11 `prefix_cache`

```json
"prefix_cache": {
  "capacity":      0,
  "in_use":        0,
  "lifetime_hits": 0
}
```

The inline prefix cache (system-prompt KV reuse). Same atomic /
non-strictly-consistent semantics as `full_cache` (§4.6).
`capacity = 0` means the cache is disabled.

### 4.12 `reasoning`

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

### 4.13 `sampling.capabilities`

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
§4.5 today).

### 4.14 `speculative`

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

## 5. Schema versioning

`build_info` includes `props_schema=<n>`. The integer `n` bumps
when the response shape changes in a way that breaks existing
clients.

### 5.1 Non-breaking changes (no version bump)

- Adding a new top-level section or a new field inside an existing
  section.
- Adding a new entry to `api.endpoints` or `reasoning.supported_efforts`.
- Loosening field bounds (e.g. extending an enum's allowed values).

Clients are required to ignore unknown fields. The schema version
does not bump for additive changes.

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
  "build_info": "luce-dflash v0.0.0+cpp props_schema=1",
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
    "draft_path":   "/.../dflash-draft-3.6-q8_0.gguf",
    "tokenizer_id": "qwen3"
  },
  "model_alias": "dflash",
  "model_card": {
    "source":                  "share/model_cards/qwen3.6-27b.json",
    "max_tokens":              32768,
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
