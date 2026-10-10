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
| POST | `/v1/hidden_states` | Prefill residual-stream activations | ✅ qwen35 |

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
| `reasoning` | object | — | Reasoning effort control (`{"effort":"medium"}`) | ✅ |
| `chat_template_kwargs` | object | — | Direct template control (`{"enable_thinking":true}`). Jinja templates (e.g. qwen4exp) also accept `preserve_thinking` (bool): whether earlier assistant turns replay their recorded `reasoning_content` inside `<think>...</think>` (default: template's own default, typically true) | ✅ |
| `stop` | string/array | — | Stop sequences | ✅ |
| `n` | int | — | Number of completions | ❌ TODO |
| `logprobs` | bool | `false` | Return log probabilities (non-streaming, qwen35 targets; see [Logprobs](#logprobs)) | ✅ |
| `top_logprobs` | int | 0 | Number of top logprobs per token (0–20; ignored without `logprobs: true`) | ✅ |
| `response_format` | object | — | JSON mode / structured output | ❌ TODO |
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
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
| `choices[].logprobs` | ✅ when `logprobs: true` |

### Logprobs

`logprobs: true` returns, for every generated token, its log-probability and the `top_logprobs` most likely alternatives in the OpenAI shape. Values come from the raw logits row the token was chosen from: temperature, top-k/top-p and penalties change which token is picked, not the reported distribution. A token that is only part of a UTF-8 character has its exact bytes in `bytes` and U+FFFD in `token`.

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Rate this task: rename a variable"}],
       "max_tokens":1,"temperature":0,"logprobs":true,"top_logprobs":3}'
```

```json
{
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "easy"},
    "finish_reason": "length",
    "logprobs": {"content": [{
      "token": "easy", "logprob": -0.0312, "bytes": [101, 97, 115, 121],
      "top_logprobs": [
        {"token": "easy", "logprob": -0.0312, "bytes": [101, 97, 115, 121]},
        {"token": "medium", "logprob": -3.52, "bytes": [109, 101, 100, 105, 117, 109]},
        {"token": "hard", "logprob": -6.87, "bytes": [104, 97, 114, 100]}
      ]
    }]}
  }]
}
```

Only `logprobs: true` on a non-streaming `/v1/chat/completions` request served by the single-device qwen35 backend returns logprobs. Elsewhere the field is ignored and the response carries none, as before logprobs existed: `stream: true`, other endpoints (including the legacy completions integer `logprobs`), a `top_logprobs` without `logprobs: true`, and other backends (layer-split, qwen35moe, DeepSeek4, Laguna, Gemma4 and Qwen3 targets). With `logprobs: true`, a `top_logprobs` that is not an integer in 0–20 returns HTTP 400. A logprobs request decodes without speculation, and on the single-request path it bypasses the prefix caches; with `--max-concurrency` it still uses prefix checkpoints, which always leave the last prompt token to prefill.

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
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
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
| `text` | object | — | Structured output / JSON schema | ❌ TODO 🔴 |
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

## POST `/v1/hidden_states` (prefill activations)

Runs the prompt through **prefill only** on the named model and returns the
residual stream (pre-final-norm, f32) after chosen blocks, at the last prompt
token and optionally mean-pooled over every prompt token. Nothing is decoded
or sampled. Use it to read a model's internal representation of a prompt,
for example as features for a separate classifier.

```bash
curl -s http://127.0.0.1:8080/v1/hidden_states -H 'Content-Type: application/json' -d '{
  "model": "qwen3.5-2b",
  "messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
  "chat_template_kwargs": {"enable_thinking": false},
  "layers": [11, 17, -1],
  "pooling": "both",
  "generation_prompt": "bare"
}'
```

```json
{
  "object": "hidden_states", "model": "qwen3.5-2b",
  "n_layers": 24, "n_embd": 2048, "prompt_tokens": 19,
  "pooling": "both", "generation_prompt": "bare", "last_token": "\n",
  "layers": {
    "11": {"last": [0.013, -0.402, ...], "mean": [0.021, -0.117, ...]},
    "17": {"last": [...], "mean": [...]},
    "-1": {"last": [...], "mean": [...]}
  },
  "timings": {"prefill_ms": 31.4}
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Model block name (`--model-name`). Required on a multi-model listener; an unknown name is 404 even with `--unknown-model primary`. |
| `messages` | array | — | OpenAI chat messages, rendered with the model's chat template like `/v1/chat/completions` (tools, reasoning and `chat_template_kwargs` apply the same way). |
| `layers` | int[] | — | 0-based block indices; entry `L` is the residual stream **after** block `L`. Negative counts from the end (`-1` = last block). HF `output_hidden_states=True` has `hidden_states[L+1]` for block `L`. |
| `pooling` | string | `"last"` | `"last"`, `"mean"` or `"both"`. `last` is always returned; `mean` (over all prompt tokens) only for `mean`/`both`. |
| `generation_prompt` | string | `"template"` | `"template"` keeps whatever the template appends after the assistant header. `"bare"` ends the prompt exactly at `<\|im_start\|>assistant\n`, dropping e.g. Qwen3.5's `<think>\n\n</think>\n\n` (thinking off) or `<think>\n` (thinking on), so the features do not depend on the thinking mode of a later request. |
| `cache_prefix` | bool | `false` | Also store this prefill in the prefix cache, so a following `/v1/chat/completions` on the same model whose prompt extends this one (typically the `"bare"` prompt plus the template's think block) restores it instead of prefilling it again. See below. |

Response `layers` is keyed by each layer exactly as requested. `last_token` is
the decoded text of the final prompt token (where `last` was read).
`timings.prefill_ms` is wall time from admission to the end of prefill.
With `cache_prefix: true` the response also carries
`"cache_prefix": {"cached_tokens": N}`: how many leading tokens of this prompt
the prefix cache holds once the request is done (`prompt_tokens` when the
checkpoint was stored or already present, 0 when it could not be stored).

Behavior:

- Always a full prefill from position 0: prefix-cache snapshots are never
  restored, and PFlash compression never applies. By default nothing is
  captured either.
- `cache_prefix: true` (opt-in) captures one prefix-cache checkpoint at the
  **end** of the prompt, through the same machinery chat requests use: an
  engine prefix checkpoint (paged K/V copy plus DeltaNet/conv recurrent
  state) under `--max-concurrency` > 1, or an end-of-prefill inline snapshot
  on the single-request worker. The cut is the prompt end, where the last
  prefill chunk stops anyway, so the readout is computed exactly as without
  the flag. A later chat request whose tokens extend this prompt finds the
  entry by exact token prefix and reports the hit in `timings`
  (`cache_hit`, `cached_prefix_tokens`). Storing is best effort: it is
  skipped when the prefix cache is disabled (`--prefix-cache-slots 0`,
  single-sequence `--paged-attention`), when another capture is in flight,
  or when the entry would not fit the resident budget. Entries age out like
  any other prefix-cache entry. The restored chat prefills its suffix in a
  different chunk split than a cold prefill would, so greedy output matches
  a no-cache run only up to the same floating-point tolerance as any other
  prefix-cache hit.
- Runs through the model's own queue and capacity like a completion. With
  `--max-concurrency` > 1 the request takes one engine slot and its prompt
  chunks are batched with other slots' prefill/decode; the readout is a
  per-slot row range of the shared step graph. Single-request models run it
  on the worker between generations.
- Cost: one `[n_embd]` row copy per requested layer per prefill chunk, plus a
  transpose + row sum when `mean` is requested. The LM head still runs for
  the last row only.

Errors (400): missing/empty/non-integer or repeated `layers`, a layer outside
the model, unknown `pooling`/`generation_prompt`, empty `messages`, image
input, `generation_prompt: "bare"` with a template that does not end in an
open `<|im_start|>assistant` turn, an unnamed model on a multi-model listener,
and backends without support. Only the single-device dense qwen35 backend
(no tensor parallelism, no KVFlash) supports it; layer-split, qwen35moe and
other architectures return 400. Unknown model under name routing: 404.

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
| **`text` (structured output)** | Codex (Responses API) | JSON schema output formatting (`{"format":{"type":"json_schema","schema":{...}}}`). Needed for structured tool outputs. |
| **`service_tier`** | Codex (Responses API) | Routing hint (e.g., `"default"`). Accept and ignore. |

### 🟡 Medium Priority

| Feature | Used By | Notes |
|---------|---------|-------|
| **`response_format`** | Chat Completions API | JSON mode / structured output (OpenAI Chat format). |
| **`metadata`** | Claude Code (Anthropic) | Request metadata for tracing. Accept and ignore. |
| **`stream_options`** | Some OpenAI clients | `{"include_usage": true}` — usage in final streaming chunk. |
| **Input validation** | — | Clamp penalty ranges [-2,2], reject invalid params with 400. |
| **`previous_response_id`** | Codex (Responses API) | Multi-turn response chaining (we handle via `input` already). |

### 🟢 Low Priority

| Feature | Notes |
|---------|-------|
| **Streaming `logprobs`** | Non-streaming chat completions return logprobs; SSE chunks do not carry them yet. |
| **`n` (multiple completions)** | Generate N choices per request. No known agent uses this. |
| **`logit_bias`** | Per-token logit adjustments. |
| **`user`** | End-user identifier (tracking only). |
| **`prompt_cache_key`** | Codex sends this for server-side caching hints. We have our own prefix cache. |
