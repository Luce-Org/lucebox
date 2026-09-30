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
| POST | `/v1/systemone` | Direct-logit typed decisions | ✅ Qwen3.5/Qwen3.6 |

---

## POST `/v1/systemone`

Runs Jev-like typed decisions without decoding. Each question is rendered as a
fixed task/criteria/state prompt, the backend performs one prefill, and the
server reads only the allowed label logits at the final `Answer:` position.

One question therefore uses one prefill pass. Multiple questions sharing one
state use one independent prefill pass per question; answers are independent
and no generated label token is appended.

```json
{
  "model": "luce-dflash",
  "state": {
    "subject": "Duplicate charge",
    "message": "Please refund the duplicate charge."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoices, and refunds",
        "technical": "Bugs and integrations"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the customer request a refund?"
    }
  }
}
```

The endpoint supports:

- `noul`, using restricted `Y`/`N` logits.
- `choice`, with 2–10 named options using `A`–`J`.
- `score`, with 2–10 ordered levels using `A`–`J`.
- String, object, or array values for `state`, `instructions`, and descriptions.

```json
{
  "model": "luce-dflash",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.91,
        "technical": 0.09
      },
      "confidence": 0.56
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.98
    }
  },
  "usage": {
    "input_tokens": 236,
    "output_tokens": 0
  }
}
```

`confidence` is `1 - normalized_entropy(probabilities)`. The server validates
that every configured answer label is one distinct tokenizer token in the
exact `Answer:` context. Invalid requests return `422`.

The fixed prompt presents `noul` labels as `Yes` and `No`, then instructs the
model to answer with exactly one criterion label. This keeps the answer token
generation-free while improving direct-logit classification with instruction
models.

The current final-logit implementation is available on the monolithic
Qwen3.5/Qwen3.6 backend. It is rejected when `--max-concurrency` enables the
concurrent-slot engine; other backends return `503` until they expose their
final-position prefill logits.

Run the sourced live quality suite against a local model with:

```bash
python3 server/tests/test_systemone_quality.py \
  --launch server/models/Qwen3.6-27B-Q4_K_M.gguf \
  --server-bin server/build/dflash_server
```

---

## Agent Turn Cache

Start the server with `--agent-turn-cache` to extend the existing in-memory
prefix cache through generated tool calls. After the model emits a valid tool
call, the server reuses its deepest compatible prefix checkpoint, replays the
uncached tail once, and saves the canonical completed turn. On the next OpenAI
Chat Completions or Responses request, only the appended tool result and new
suffix need prefill.

This is a server-wide optimization; request bodies do not change. It requires
`--prefix-cache-slots` to be nonzero. Compressed or token-rewritten prompts and
requests without a compatible checkpoint safely fall back to ordinary prefix
caching.

The replay moves prefill work out of the follow-up request when tool execution
is long enough to overlap it; it does not eliminate that work. Paged attention
and `--max-concurrency` do not yet support shared prefix blocks, so they cannot
be combined with Agent Turn Cache.

The exact full-prompt cache may remain enabled for identical-request hits.
Disable it only when benchmarking the incremental Agent Turn Cache benefit.

Successful Chat Completions and Responses requests expose the measured result
under `usage.timings`:

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
| `chat_template_kwargs` | object | — | Direct template control (`{"enable_thinking":true}`) | ✅ |
| `stop` | string/array | — | Stop sequences | ✅ |
| `n` | int | — | Number of completions | ❌ TODO |
| `logprobs` | bool | — | Return log probabilities | ❌ TODO |
| `top_logprobs` | int | — | Number of top logprobs per token | ❌ TODO |
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
| **`logprobs` / `top_logprobs`** | Token probabilities in response. Debugging/analysis only. |
| **`n` (multiple completions)** | Generate N choices per request. No known agent uses this. |
| **`logit_bias`** | Per-token logit adjustments. |
| **`user`** | End-user identifier (tracking only). |
| **`prompt_cache_key`** | Codex sends this for server-side caching hints. We have our own prefix cache. |
