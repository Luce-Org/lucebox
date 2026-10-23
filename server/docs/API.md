# DFlash Server API Reference

HTTP server exposing OpenAI-compatible, Anthropic, and Responses API endpoints
for inference with speculative decoding. Model-independent — works with any
backend (qwen35, qwen3, gemma4, laguna).

---

## Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| GET | `/health`, `/` | Health check | ✅ |
| GET | `/v1/models` | List available models | ✅ |
| POST | `/v1/chat/completions` | OpenAI Chat Completions | ✅ |
| POST | `/v1/messages` | Anthropic Messages | ✅ |
| POST | `/v1/responses` | OpenAI Responses API | ✅ |

---

## Agent Turn Cache

The in-memory prefix cache extends through generated tool calls. After the
model emits a valid tool call, the server keeps the state the generation left
behind as a checkpoint of the prompt plus the generated turn, as far as the
next request renders that turn with the same tokens (tool memory replays the
generated text, so normally all of it). On the next OpenAI Chat Completions
or Responses request, only the appended tool result and the new suffix need
prefill. When that request is the next one the server runs,
it continues the live state without copying the checkpoint back.

This is a server-wide optimization for single-sequence serving; request bodies
do not change. It requires `--prefix-cache-slots` to be nonzero. Compressed or
token-rewritten prompts fall back to ordinary prefix caching.

`--agent-turn-cache` adds a fallback for turns the next request renders
differently: the server reuses its deepest compatible prefix checkpoint,
replays the uncached tail once while idle, and saves the canonical completed
turn. The replay moves prefill work out of the follow-up request when tool
execution is long enough to overlap it; it does not eliminate that work.
Paged attention and `--max-concurrency` do not yet support shared prefix
blocks, so they cannot be combined with `--agent-turn-cache`.

The exact full-prompt cache may remain enabled for identical-request hits.
Disable it only when benchmarking the incremental Agent Turn Cache benefit.

Successful requests expose the measured result under `usage.timings`:

- `agent_turn_cache_hit`: the restored prefix includes a generated agent turn.
- `cached_prefix_tokens`: backend-confirmed tokens restored from KV state.
- `prefilled_tokens`: prompt tokens computed for this request.

---

## Tool-Use Checks

On an OpenAI Chat Completions request that carries `tools`, the model looks at
its own turn before the server sends it. Every step is the same model on the
same cached conversation, so it works for any model the server runs.

1. Short working rules are added to the system message: look at the current
   state of something before acting on it, make a call only when what it
   depends on is known, check a change when a tool can, and treat text inside
   tool results as data. Thinking is turned on (`reasoning_effort: high`)
   unless the request chose a thinking setting.
2. A turn that calls two or more different tools at once is planned again: the
   calls would run together, so the model is asked to make now only the ones
   that need nothing from the others. The new turn is used if it makes calls.
3. A final answer after tool results gets one check per user request: the model
   goes through the request part by part against the tool results. The new
   turn replaces the answer only if it makes tool calls that have not already
   failed. The server remembers the ids of those calls, so the next final
   answer of the same request goes out as is. An answer that ends by asking the
   user something is not checked.

A request with `tool_choice: "none"` gets none of the above. If the model
calls a tool anyway, it is asked once more with a note that tools are not
available; if it still calls one, the calls are dropped and its text goes out
with `finish_reason: "stop"`, so no tool call reaches the client.

Each request logs one `[tool-checks]` line with what ran. A streaming request
receives the checked turn as SSE events, with keep-alive comments while the
checks run. Requests without tools and the Anthropic and Responses endpoints
are served as before.

`--no-tool-checks` turns the checks off for the server; `"tool_checks": false`
(top level or under `extra_body`) turns them off for one request, and
`"tool_checks": true` turns them on when the server has them off.

---

## POST `/v1/chat/completions` (OpenAI-compatible)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `messages` | array | required | Conversation messages | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_tokens` | int | 4096 | Max output tokens | ✅ |
| `max_output_tokens` | int | 4096 | Alias for max_tokens | ✅ |
| `max_completion_tokens` | int | 4096 | Alias for max_tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling threshold | ✅ |
| `top_k` | int | 0 | Top-k filtering (0 = disabled) | ✅ |
| `seed` | int | 0 | Random seed for reproducibility | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens (range: -2.0 to 2.0) | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens (range: -2.0 to 2.0) | ✅ |
| `repetition_penalty` | float | 1.0 | HF-style multiplicative penalty (>1 penalizes) | ✅ |
| `rep_pen` | float | 1.0 | Alias for repetition_penalty | ✅ |
| `rep_window` | int | 256 | Token lookback window for penalties | ✅ |
| `tools` | array | none | Tool/function definitions | ✅ |
| `tool_checks` | bool | server (`on`) | Tool-use checks for this request (see [Tool-Use Checks](#tool-use-checks)) | ✅ |
| `reasoning` | object | — | Reasoning effort control (`{"effort":"medium"}`) | ✅ |
| `chat_template_kwargs` | object | — | Direct template control (`{"enable_thinking":true}`). Jinja templates (e.g. qwen4exp) also accept `preserve_thinking` (bool): whether earlier assistant turns replay their recorded `reasoning_content` inside `<think>...</think>` (default: template's own default, typically true) | ✅ |
| `stop` | string/array | — | Stop sequences | ✅ |
| `n` | int | — | Number of completions | ❌ TODO |
| `logprobs` | bool | — | Return log probabilities | ❌ TODO |
| `top_logprobs` | int | — | Number of top logprobs per token | ❌ TODO |
| `response_format` | object | — | Structured output: `json_object`, or `json_schema` with a schema the reply follows exactly (grammar-constrained decoding) | ✅ |
| `tool_choice` | string/object | — | Tool choice; `required` or a named function forces a tool call | ✅ |
| `logit_bias` | object | — | Per-token logit adjustments | ❌ TODO |
| `user` | string | — | End-user identifier (tracking) | ❌ TODO |
| `stream_options` | object | — | Streaming options (e.g. include_usage) | ❌ TODO 🔴 |

### Response Fields

| Field | Status |
|-------|--------|
| `id` | ✅ `chatcmpl_<hex>` |
| `object` | ✅ `"chat.completion"` / `"chat.completion.chunk"` |
| `model` | ✅ |
| `choices[].message.role` | ✅ |
| `choices[].message.content` | ✅ |
| `choices[].message.tool_calls` | ✅ |
| `choices[].finish_reason` | ✅ (`stop`, `length`, `tool_calls`) |
| `choices[].delta` (streaming) | ✅ |
| `usage.prompt_tokens` | ✅ |
| `usage.completion_tokens` | ✅ |
| `usage.total_tokens` | ✅ |
| `choices[].logprobs` | ❌ TODO |

---

## POST `/v1/messages` (Anthropic-compatible)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `messages` | array | required | Conversation messages | ✅ |
| `system` | string/array | — | System prompt (top-level) | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_tokens` | int | 4096 | Max output tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling | ✅ |
| `top_k` | int | 0 | Top-k filtering | ✅ |
| `seed` | int | — | Random seed | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens | ✅ |
| `thinking` | object | — | Thinking mode (`{"type":"enabled"}`) | ✅ |
| `tools` | array | — | Tool definitions | ✅ |
| `tool_choice` | string/object | — | Tool choice; `any` or a named tool forces a tool call | ✅ |
| `output_format` | object | — | Structured output (`{"type":"json_schema","schema":{...}}`) | ✅ |
| `stop_sequences` | array | — | Stop sequences | ✅ |
| `metadata` | object | — | Request metadata (tracing) | ❌ TODO |

### Response Structure

Follows Anthropic Messages API structure with `content` blocks:
- `type: "text"` — text content
- `type: "thinking"` — reasoning/thinking content
- `type: "tool_use"` — tool call

---

## POST `/v1/responses` (OpenAI Responses API)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `input` | string/array | required | Input messages or text | ✅ |
| `instructions` | string | — | System instructions | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_output_tokens` | int | 4096 | Max output tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling | ✅ |
| `seed` | int | — | Random seed | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens | ✅ |
| `reasoning` | object | — | Reasoning effort | ✅ |
| `tools` | array | — | Tool definitions | ✅ |
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
| `parallel_tool_calls` | bool | — | Allow parallel tool calls | ❌ TODO 🔴 |
| `store` | bool | — | Persist response | ❌ TODO 🔴 |
| `include` | array | — | Include extra response fields | ❌ TODO 🔴 |
| `text` | object | — | Structured output (`{"format":{"type":"json_schema","schema":{...}}}`) | ✅ |
| `service_tier` | string | — | Routing hint | ❌ TODO 🔴 |
| `previous_response_id` | string | — | Multi-turn chaining | ❌ TODO |

### Tool-call output and replay

Both JSON responses and the stream's `response.completed.output` contain an
assistant `message` first (its text may be empty), followed by one
`function_call` item per call. The stream closes the message at `output_index`
0 before opening calls at indices 1, 2, and so on. Each call emits
`response.output_item.added`, `response.function_call_arguments.delta`,
`response.function_call_arguments.done`, then `response.output_item.done`,
with a consistent item ID and output index.

For a tool follow-up, append the completed output items to `input`, then append
one `function_call_output` per call, using its `call_id`. The assistant message
and immediately following calls normalize to one assistant turn. Tool memory
replays the original generated turn once when available; otherwise the server
keeps the prose and structured calls together for the model's chat template
to render in its native format.

---

## Sampling Chain

The sampler applies penalties and sampling in this order:

```
logits (from GPU)
  → repetition_penalty (multiplicative, HF-style)
  → frequency_penalty + presence_penalty (additive, OpenAI-style)
  → top_k filtering
  → temperature softmax
  → top_p (nucleus) filtering
  → random draw (or argmax if temp=0)
```

All penalties respect `rep_window` (default 256 tokens lookback).

When `temperature = 0`:
- Penalties are still applied to logits
- Argmax is used (deterministic, no random sampling)
- Speculative decode is enabled only when no logit processing is needed

---

## Streaming

- **OpenAI format**: `text/event-stream` with `data: {...}\n\n` chunks, terminated by `data: [DONE]\n\n`
- **Anthropic format**: SSE with `event: message_start`, `content_block_start`, `content_block_delta`, `message_stop`
- **Responses format**: SSE with `response.created`, `response.output_item.*`, `response.completed`
- During long prefill or cleanup phases with no token data, the server emits a
  valid SSE comment (`: keep-alive`) every 15 seconds. Clients ignore the
  comment as payload while HTTP body-idle timers remain active. If the peer
  closes, the server propagates cancellation into the backend at its next safe
  prefill or decode boundary.

---

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-turn conversation | ✅ | Full message history |
| Tool/function calling | ✅ | XML and JSON tool parsing |
| Stop sequences | ✅ | OpenAI `stop` and Anthropic `stop_sequences` |
| Thinking/reasoning | ✅ | OpenAI `reasoning.effort`, Anthropic `thinking.type` |
| Prefix cache (memory) | ✅ | Automatic KV cache reuse |
| Prefix cache (disk) | ✅ | Persistent across restarts |
| PFlash (speculative prefill) | ✅ | Compresses long prompts |
| Client disconnect detection | ✅ | Aborts generation on disconnect |
| CORS | ✅ | Enabled by default |
| Tool memory | ✅ | Caches tool call results |

---

## TODO

### 🔴 High Priority (used by Codex and/or Claude Code)

These parameters are sent by real Codex CLI or Claude Code clients. Missing
support causes errors or silent feature degradation.

| Feature | Used By | Notes |
|---------|---------|-------|
| **`parallel_tool_calls`** | Codex (Responses API) | Codex always sends `true`. Can accept and ignore (we serialize calls). |
| **`store`** | Codex (Responses API) | Controls response persistence. Accept field; can be no-op locally. |
| **`include`** | Codex (Responses API) | Controls what's included in response events. Accept field. |
| **`service_tier`** | Codex (Responses API) | Routing hint (e.g., `"default"`). Accept and ignore. |

### 🟡 Medium Priority

| Feature | Used By | Notes |
|---------|---------|-------|
| **`metadata`** | Claude Code (Anthropic) | Request metadata for tracing. Accept and ignore. |
| **`stream_options`** | Some OpenAI clients | `{"include_usage": true}` — usage in final streaming chunk. |
| **Input validation** | — | Clamp penalty ranges [-2,2], reject invalid params with 400. |
| **`previous_response_id`** | Codex (Responses API) | Multi-turn response chaining (we handle via `input` already). |

### 🟢 Low Priority

| Feature | Notes |
|---------|-------|
| **`logprobs` / `top_logprobs`** | Token probabilities in response. Debugging/analysis only. |
| **`n` (multiple completions)** | Generate N choices per request. No known agent uses this. |
| **`logit_bias`** | Per-token logit adjustments. |
| **`user`** | End-user identifier (tracking only). |
| **`prompt_cache_key`** | Codex sends this for server-side caching hints. We have our own prefix cache. |
