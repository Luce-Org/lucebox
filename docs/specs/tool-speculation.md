# Model-agnostic tool speculation

Tool speculation lets an external service predict and start one safe tool
before local model execution. The model remains authoritative: the engine
returns the private result only when the generated function name and canonical
JSON arguments match the prediction exactly.

The engine does not load or identify the predictor model. Qwen, another model,
rules, or a cached predictor can implement the same protocol. Hardware and CPU
affinity also belong to the service, keeping DS4, DSpark, and autoregressive
decoding unchanged.

## Configuration

```text
--tool-spec-endpoint http://127.0.0.1:19090/v1/speculate
--tool-spec-allow get_weather
--tool-spec-allow search_documents
```

Only read-only or idempotent tools should be allowlisted. Optional controls:

```text
--tool-spec-key <bearer-token>
--tool-spec-min-confidence 0.75
--tool-spec-start-timeout-ms 2000
--tool-spec-finish-timeout-ms 60000
```

The predictor runs before local prompt preparation and decoding. A successful
`start` response therefore means the tool is already running when model compute
begins. Predictor failure only disables speculation for that request.

## Service protocol

Every request and response is JSON. The current protocol identifier is
`dflash.tool-speculation.v1`.

The engine starts an attempt with the normalized conversation and only the
allowlisted tool definitions:

```json
{
  "protocol": "dflash.tool-speculation.v1",
  "operation": "start",
  "request_id": "chatcmpl_...",
  "messages": [{"role": "user", "content": "Weather in Rome"}],
  "tools": [{"name": "get_weather", "parameters": {"type": "object"}}],
  "tool_choice": "auto",
  "min_confidence": 0.75
}
```

The service predicts a call, starts it privately, then responds:

```json
{
  "ticket": "opaque-service-ticket",
  "call": {
    "name": "get_weather",
    "arguments": {"city": "Rome", "unit": "celsius"}
  },
  "confidence": 0.91
}
```

The engine rejects unknown tools, malformed arguments, and predictions below
its threshold. Rejected tickets receive `cancel`.

After generation, one exact call receives `commit`:

```json
{
  "protocol": "dflash.tool-speculation.v1",
  "operation": "commit",
  "request_id": "chatcmpl_...",
  "ticket": "opaque-service-ticket"
}
```

The service waits for the already-running tool if necessary and returns:

```json
{"ok": true, "result": {"temperature": 24}}
```

A wrong prediction, generation failure, or disconnect receives `cancel` with a
reason. The service should acknowledge cancellation promptly and must never
expose a private result through that response. It must also expire abandoned
tickets, because a timeout can prevent the engine from delivering `commit` or
`cancel` reliably.

Non-streaming responses expose the outcome under
`dflash_tool_speculation`. Streaming responses emit one API-shaped extension
immediately before the normal terminal event. On a miss, this metadata contains
no `result` field. A compatible client uses `result` as the completed output for
`call_id` only when `status` is `hit`, and must not invoke that call a second
time. For `miss`, `deferred`, `cancelled`, or `failed`, it follows the normal
tool-execution path. Enable the server feature only for clients that understand
this additive extension.
