# harness/routing

## lucerouter

`lucerouter/` is an OpenAI-compatible gateway that sends each chat request to
one of the models in `config/backends.json`. It records every decision so we
can compare router approaches.

```bash
cd harness/routing
uv sync
uv run pytest                      # mock backends only, no GPU
uv run python -m lucerouter.gateway --config config/backends.json \
    --router "load_aware>brick:service=brick-max" \
    --port 8400 --log runs/gateway/decisions.jsonl
```

Clients point at `http://127.0.0.1:8400/v1`:

- If `model` is `"route"`, `"auto"` or missing, the router picks the model.
- If `model` names a configured model or its `served_name`, the gateway passes the request through without routing.
- The gateway rewrites `model` to the backend's `served_name`, or to the config key when `served_name` is not set.
- It forwards every other field as-is, including `chat_template_kwargs` from the thinking router.
- It drops only the gateway's own `route` field.
- Streaming responses are relayed byte for byte.

Responses carry the `x-lucerouter-model`, `x-lucerouter-router` and
`x-lucerouter-request-id` headers. Non-streaming JSON responses also get a
top-level `"lucerouter": {decision}` field.

Other endpoints:

- `GET /v1/models` lists `route` and then every backend.
- `GET /health`
- `GET /router/stats` reports request counts, counts per model and the average router latency.

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
| `fixed` | leaf | `model` |
| `brick` | leaf | `service=brick-max`, `t_small=0.8`, `t_mid=0.5`, `h_max=0.5`, `small`/`mid`/`large` (by rank; `small=none` drops the 0.8b tier), `no_think` (llama-server kind only), `on_error=<large>` |
| `load_aware` | wrapper | `margin_up=1.0`, `margin_down=0.1`, `promote=0` (off) |

How each router decides:

- **brick** calls the Brick classifier.
  - On luce_server, it sends `/v1/chat/completions` with `logprobs`, `top_logprobs=20`, `max_tokens=1` and thinking off.
  - On llama-server, it sends a raw ChatML `/completion` instead.
  - It then applies these rules in order:
    - `p_easy >= t_small` sends the request to the small model.
    - `p_easy + p_medium >= t_mid` together with `p_hard < h_max` sends it to the mid model.
    - Anything else goes to the large model.
- **load_aware** reads backend load from `/status/json`, cached for 200 ms.
  - When the chosen model is full, it moves one rank up if that model is free. If not, it moves one rank down when the inner margin is at most `margin_down`.
  - With `promote > 0`, a borderline decision (margin below `promote`) goes to the largest model while that model is idle.
  - When load is unknown, the decision stays as it is.

**Escalation (swarm).** A router may return a target that is not in the config,
such as `"swarm"`. The gateway passes such targets to
`create_app(..., extra_targets={"swarm": handler})`. Without a handler it
returns 502.

### Decision log

The gateway appends one JSON line per request:

```json
{"ts": "2026-09-30T11:09:51+00:00", "request_id": "…", "session_id": "s1",
 "routed": true, "stream": false, "status": 200,
 "decision": {"model": "qwen35-2b", "router": "brick:service=brick-max",
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
- Requests that passed through without routing have `routed: false` and router `"passthrough"`.

### Config fields read by lucerouter

`config/backends.json`:

- `models.<name>`: `base_url` (ending in `/v1`), `kind` (`luce_server` or `llama-server`), `rank` (0 is the smallest), and optionally `served_name`, `context` (tokens) and `capacity` (the concurrency budget used when `/status/json` does not report one; default 1).
- `services.<name>`: `base_url`, `kind`, and optionally `served_name` and `prompt_suffix` (llama-server Brick only).

Load is read from `<root>/status/json` on luce_server, or from `/slots` and then `/metrics` on llama-server. `<root>` is `base_url` without `/v1`.
