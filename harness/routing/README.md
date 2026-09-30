# harness/routing

## lucerouter

`lucerouter/` is an OpenAI-compatible gateway that sends each chat request to
one of the *arms* in `config/backends.json`. It records every decision so we
can compare router approaches.

An arm is a backend model plus request overrides. The default arms are:

| arm | backend | thinking |
|---|---|---|
| `qwen35-2b` | qwen35-2b | off |
| `qwen38-27b` | qwen38-27b | off |
| `qwen38-27b-think` | qwen38-27b | on |

The two 27B arms share one backend and one queue. `qwen35-0.8b` is still a
configured model, so you can pass requests through to it or add it as an arm,
but it is not a routing candidate by default. Without an `arms` section every
model is one arm and the request's thinking setting is left alone.

```bash
cd harness/routing
uv sync                            # add --extra brick for the brick_skill router (CPU torch + transformers)
uv run pytest                      # mock backends only, no GPU
uv run python -m lucerouter.gateway --config config/backends.json \
    --router "load_aware>brick:service=brick-max" \
    --port 8400 --log runs/gateway/decisions.jsonl
```

Clients point at `http://127.0.0.1:8400/v1`:

- If `model` is `"route"`, `"auto"` or missing, the router picks an arm. The gateway sends the request to the arm's backend. When the arm sets `thinking`, the gateway also sets `chat_template_kwargs.enable_thinking` to that value, replacing whatever the client sent; other `chat_template_kwargs` keys are kept.
- If `model` names a configured model or its `served_name`, the gateway passes the request through untouched, without routing.
- If `model` names an arm that is not also a model name (for example `qwen38-27b-think`), the gateway applies that arm's overrides without routing.
- The gateway rewrites `model` to the backend's `served_name`, or to the config key when `served_name` is not set.
- It forwards every other field as-is.
- It drops only the gateway's own `route` field.
- Streaming responses are relayed byte for byte.

Responses carry the `x-lucerouter-model` (the arm), `x-lucerouter-backend`,
`x-lucerouter-router` and `x-lucerouter-request-id` headers. Non-streaming JSON responses also get a
top-level `"lucerouter": {decision}` field.

Other endpoints:

- `GET /v1/models` lists `route`, then every arm, then every model that is not also an arm.
- `GET /health`
- `GET /router/stats` reports request counts, counts per arm and per backend, and the average router latency.

Eval scripts can call a router directly, without the gateway:

```python
from lucerouter.config import load_config
from lucerouter.routers import build
from lucerouter.types import RouteRequest

router = build("brick:service=brick-max,t_small=0.8", load_config("config/backends.json"))
d = await router.route(RouteRequest(messages=[{"role": "user", "content": "..."}]))
await router.aclose()
```

### Router spec grammar

```
spec   := layer ( ">" layer )*       # wrappers first (outer to inner), one leaf router last
layer  := name [ ":" key=value ( "," key=value )* ]
lists  := a|b|c                      # e.g. models=qwen35-2b|qwen38-27b ; "none" = unset
```

You can also pass the spec as JSON:
`{"router": "load_aware", "params": {...}, "inner": {"router": "brick", "params": {...}}}`.

| router | kind | params (defaults) |
|---|---|---|
| `fixed` | leaf | `model` (an arm) |
| `brick` | leaf | `service=brick-max`, `t_small=0.8`, `t_mid=0.5`, `h_max=0.5`, `small`/`mid`/`large` (arms by rank; `small=none` drops the cheapest tier), `no_think` (llama-server kind only), `on_error=<large>` |
| `brick_skill` | leaf | `skills=<skill table JSON>` (required), `latency=<profile JSON>` (config `latency_defaults` fill gaps), `service=brick-max`, `r=0`, `beta` and `lam` (override `r`'s values), `tie_eps=0.03`, `arms=a\|b\|c` (candidates; default all), `live_load=on`, `capability=/home/berto/models/routing/brick-capability`, `max_length=512` |
| `load_aware` | wrapper | `margin_up=1.0`, `margin_down=0.1`, `promote=0` (off) |

How each router decides:

- **brick** calls the Brick classifier.
  - On luce_server, it sends `/v1/chat/completions` with `logprobs`, `top_logprobs=20`, `max_tokens=1` and thinking off.
  - On llama-server, it sends a raw ChatML `/completion` instead.
  - It then applies these rules in order:
    - `p_easy >= t_small` sends the request to the small arm (default `qwen35-2b`).
    - `p_easy + p_medium >= t_mid` together with `p_hard < h_max` sends it to the mid arm (default `qwen38-27b`).
    - Anything else goes to the large arm (default `qwen38-27b-think`).
- **brick_skill** is Brick's real selection rule, with latency in place of cost. It is described in the next section.
- **load_aware** reads backend load from `/status/json`, cached for 200 ms. Load is per backend model, so both 27B arms see the same queue.
  - When the chosen arm's backend is full, it moves to the nearest higher arm whose backend is free (when the inner margin is at most `margin_up`). Failing that, it moves to the nearest lower arm with a free backend when the margin is at most `margin_down`.
  - With `promote > 0`, a borderline decision (margin below `promote`) goes to the top arm while that arm's backend is idle.
  - When load is unknown, the decision stays as it is.
  - `brick_skill` already prices the live queue into its latency term, so load awareness is built in. On top of it, `load_aware` only acts on full backends and close J gaps.

### brick_skill: Brick's selection rule with latency instead of cost

This follows "Brick: Spatial Capability Routing for the MoM Paradigm"
([arXiv 2606.13241](https://arxiv.org/html/2606.13241), sections 7.3 to 7.6 and Table 6) and
its reference router ([regolo-ai/brick-SR1](https://github.com/regolo-ai/brick-SR1),
`apps/router/src/spatial-router/pkg/brickrouting/router.go`). The code is in
`lucerouter/brick_math.py`.

Two signals feed it:

- **Capability** p(x): six probabilities over `coding`, `creative_synthesis`,
  `instruction_following`, `math_reasoning`, `planning_agentic` and `world_knowledge`. They come
  from [`regolo/brick-modernbert-capability-classifier`](https://huggingface.co/regolo/brick-modernbert-capability-classifier)
  (Apache-2.0), fed all user message texts, truncated to 512 tokens. As the Brick runtime does,
  the router applies softmax and then L1 normalisation.
- **Complexity**: the `brick` service's easy/medium/hard probs. Their argmax gives the label, and
  its probability gives the confidence c. When the service fails, the router uses medium with c = 1.

For each candidate arm m:

```
tau_q = c * tau[label] + (1 - c) * tau[medium]     tau = easy .55, medium .72, hard .88
z_q   = b + mu * logit(tau_q)                      required level
D_m   = sqrt( sum_c max(0, p_c z_q - p_c logit s_mc)^2 + lam * sum_c max(0, p_c logit s_mc - p_c z_q)^2 )
J_m   = D_m + beta * Lat_m(x) / max_k Lat_k(x)     choose argmin J
```

- s_mc are the arm's skill values, clipped to [0.02, 0.98].
- Under-capacity has weight 1. Over-capacity has weight lam, which is small, so an arm that is
  more capable than needed costs little but is not free.
- The knob `r` in [-1, 1] sets mu, b, beta and lam together, using Brick's locked constants.
  At r = 0 they are mu .345, b .822, beta .231 and lam .045. `beta=` and `lam=` override single
  values.
- J values less than `tie_eps` (0.03) apart are ties. A tie goes to the arm with the higher
  expected success sum_c p_c s_mc, and then to the lower latency.
- Brick divides the model's price by the pool's maximum price. We divide latency by the maximum
  over the candidate arms in the same way, so the term lies in (0, 1] and beta keeps Brick's scale.
- When the capability classifier fails, the complexity label picks the arm: easy goes to
  the cheapest arm, medium to the middle one and hard to the top one. This mirrors
  Brick's `model_map` fallback.

**Latency term** (`lucerouter/latency.py`):

```
Lat_m(x) = queue_m + ttft_ms + ttft_ms_per_token * prompt_tokens + tpot_ms * E[output tokens]
```

- `prompt_tokens` is the number of prompt characters divided by the profile's `chars_per_token`.
- E[output tokens] mixes the per-class `out_tokens` with the complexity probs and is capped at
  the request's `max_tokens`. For the think arm it includes the reasoning tokens.
- `queue_m` = in_flight / capacity x `service_ms` of the arm's backend, read from the live
  LoadMonitor. It is 0 when load is unknown or `live_load=off`.

Both the profile and the skill table are fitted from an eval run (see
[data/README.md](data/README.md)):

```bash
uv run python -m eval.fit_latency --run pilot                 # -> runs/pilot/latency_profile.json
uv run --extra brick python -m eval.fit_skills --run pilot    # -> runs/pilot/skills.json (train split)
```

Without a profile file the router uses `latency_defaults` from `config/backends.json`. Those
numbers are hand-set placeholders.

**Capability classifier on CPU (temporary).** luce_server has no encoder architectures, so
the ModernBERT classifier runs in the gateway process with torch on CPU (`uv sync --extra
brick`). The repo publishes no ONNX export. The model loads lazily on the first request (about 2 s) and
runs on its own worker thread, so it does not block the event loop. After that a call takes
50-70 ms for short prompts and about 400 ms at 512 tokens. The complexity call and the
capability call run concurrently. The classifier is about 395M parameters, and its config
has ModernBERT-large dimensions even though the card says -base. Download it with:

```bash
uvx --from huggingface_hub hf download regolo/brick-modernbert-capability-classifier \
    --local-dir /home/berto/models/routing/brick-capability
```

Tests and other callers can inject `build(spec, config, capability_fn=fn)` in place of the
local model.

Gateway example:

```bash
uv run --extra brick python -m lucerouter.gateway --config config/backends.json \
    --router "brick_skill:service=brick-max,skills=runs/pilot/skills.json,latency=runs/pilot/latency_profile.json" \
    --port 8400 --log runs/gateway/decisions.jsonl
```

A `brick_skill` decision records the following:

- `scores`: `{arm: {D, lat_ms, a, J, success, queue_ms, out_tokens}}`.
- `signals`: `margin` (the J gap to the runner-up), `tau_q`, `z_q`, `capability`, `complexity`, `params`, `brick_ms` and `capability_ms`.

**Escalation (swarm).** A router may return a target that is not in the config,
such as `"swarm"`. The gateway passes such targets to
`create_app(..., extra_targets={"swarm": handler})`. Without a handler it
returns 502.

### Decision log

The gateway appends one JSON line per request:

```json
{"ts": "2026-09-30T11:09:51+00:00", "request_id": "…", "session_id": "s1",
 "routed": true, "stream": false, "status": 200, "arm": "qwen38-27b", "backend": "qwen38-27b",
 "decision": {"model": "qwen38-27b", "router": "brick:service=brick-max",
              "reason": "p_easy+p_medium 0.80 >= t_mid 0.5, …",
              "scores": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
              "signals": {"margin": 0.3, "brick_ms": 21.4, "service": "brick-max"},
              "latency_ms": 22.0},
 "router_latency_ms": 22.0, "backend_ttft_ms": 85.1, "backend_total_ms": 912.3,
 "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}, "error": null}
```

- `backend_*` times are measured from the moment the request is sent to the backend. The router's own time is `router_latency_ms`, which is not included in them.
- Non-streaming responses log `backend_ttft_ms == backend_total_ms`.
- Streaming usage comes from the last SSE chunk that carries `usage`, so pass `stream_options.include_usage` if the backend requires it.
- Requests that passed through without routing have `routed: false` and router `"passthrough"`. Requests naming an arm have `routed: false` and router `"arm"`.
- `arm` is the chosen route target and `decision.model` is the same value. `backend` is the model server that got the request.

### Config fields read by lucerouter

`config/backends.json`:

- `arms.<name>`: `model` (a key of `models`), `thinking` (`true`/`false`, or omit to leave the request alone) and `rank` (0 is the cheapest). When this section is absent, every model is one arm.
- `latency_defaults`: a hand-set latency profile for `brick_skill` (same shape as `eval/fit_latency.py` output).
- `models.<name>`: `base_url` (ending in `/v1`), `kind` (`luce_server` or `llama-server`), `rank` (0 is the smallest), and optionally `served_name`, `context` (tokens) and `capacity` (the concurrency budget used when `/status/json` does not report one; default 1).
- `services.<name>`: `base_url`, `kind`, and optionally `served_name` and `prompt_suffix` (llama-server Brick only).

Load is read from `<root>/status/json` on luce_server, or from `/slots` and then `/metrics` on llama-server. `<root>` is `base_url` without `/v1`.
