# Qwen3.6-27B forge benchmark: tool_result fix — bragi — 2026-05-31

Investigation of Qwen3.6 forge=0% and the root-cause fix for multi-turn
tool-calling conversations via the Anthropic Messages API.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * GPU throttled ~86–90 W / 1515 MHz (Windows Balanced mode).
* **Model**: Qwen3.6-27B-Q4_K_M, optimal config:
  ```toml
  budget = 16
  max_ctx = 98304
  cache_type_k = "tq3_0"
  cache_type_v = "tq3_0"
  fa_window = 0
  think_max = 15488
  ```
* **Images tested**:
  * `658d016f-cuda12` — pre-fix (Gemma4 call:verb fix, no tool_result fix)
  * `dc20057e-cuda12` — post-fix (adds tool_use + tool_result block handling)

## forge benchmark: pre-fix Qwen3.6 results

`uv run luce-bench --areas forge --no-think --questions 5`

| Leg | forge pass_rate | notes |
|-----|-----------------|-------|
| nothink (image `658d016f`) | 0% (0/5) | all 15 iterations, model loops |
| think (image `658d016f`) | 0% (0/5) | identical to nothink — think flag has no effect on forge |

Scenario detail (both nothink and think identical):

| # | Scenario | Result | wall | calls | notes |
|---|----------|--------|------|-------|-------|
| 1 | basic_2step | FAIL | 24.84s | 15 | model loops calling same tool |
| 2 | sequential_3step | FAIL | 32.27s | 15 | same loop pattern |
| 3 | error_recovery | FAIL | 4.56s | 3 | fast failure |
| 4 | tool_selection | FAIL | 30.25s | 15 | loop |
| 5 | argument_fidelity | FAIL | 26.66s | 15 | loop |

### Why think vs nothink are identical

The forge runner uses `forge-guardrails`' own `AnthropicClient` to send
requests. The luce-bench `--think`/`--no-think` flag injects `/think` or
`/no_think` into the prompt via the `thinking_control` model-card field, but
the forge runner doesn't go through that injection path. Both modes send
identical raw API requests; the results are deterministic.

## Root cause: `normalize_chat_messages()` drops tool_use and tool_result blocks

**File**: `server/src/server/http_server.cpp`, function `normalize_chat_messages()`

When parsing Anthropic-format messages with array content, the code only
extracted blocks with types `"text"`, `"input_text"`, `"output_text"`. The
`"tool_use"` and `"tool_result"` block types were silently dropped:

```cpp
// BEFORE FIX (lines ~511-518):
for (const auto & part : m["content"]) {
    std::string ptype = part.value("type", "");
    if (ptype == "text" || ptype == "input_text" ||
        ptype == "output_text") {
        cm.content += part.value("text", "");
    }
    // tool_use and tool_result: silently ignored
}
```

**Consequence**: In a multi-turn tool-calling conversation:
1. Turn 1: Model correctly calls `get_country_info` (server returns `tool_use` block)
2. Turn 2: Client sends back assistant message with `tool_use` block + user message
   with `tool_result` block containing the tool output
3. `normalize_chat_messages()` drops both blocks → model receives only role
   delimiters (+10 tokens), no conversation history
4. Model sees the original user question again with no tool result → calls
   `get_country_info` again → infinite loop, 15 iterations → FAIL

**Diagnostic evidence**:
- Token count: first turn = 266 tokens, second turn = 276 tokens (+10 only)
- Server log: `msgs=4 prompt_tokens=343` → `msgs=6 prompt_tokens=353` (+10)
- Expected delta for real tool_use + tool_result: ~50-100 tokens, not 10
- Manual test confirmed: model always outputs identical 27-token `get_country_info`
  call regardless of conversation depth

## Fix (commit `dc20057e`)

Two additions to `normalize_chat_messages()`:

### 1. Anthropic-format assistant tool_use blocks

When the assistant message contains `tool_use` content blocks, extract their
IDs and look them up in `tool_memory` (same as the OpenAI `tool_calls` path).
`tool_memory` stores the raw model output (the `<tool_call>...</tool_call>`
XML block) keyed by the tool call ID the server assigned when emitting it.

If the tool_memory lookup misses (cross-session replay where the server was
restarted), fall back to synthesizing the XML from the block fields:
```
<tool_call>
<function=name>
<parameter=key>value</parameter>
</function>
</tool_call>
```

### 2. Anthropic-format user tool_result blocks

When a user message content array contains `tool_result` type blocks, push
each as a separate `{"tool", content, tool_use_id}` ChatMessage. The
`chat_template.cpp` renderer wraps these in `<tool_response>...</tool_response>`
tags inside a user turn, which is exactly what Qwen3.6's chat template expects.

If all content was `tool_result` blocks (typical case — no text mixed in),
skip pushing an empty user message container.

## forge benchmark: post-fix results

`uv run luce-bench --areas forge --no-think --questions 5` on image `dc20057e-cuda12`

| Leg | forge pass_rate | notes |
|-----|-----------------|-------|
| pre-fix nothink (image `658d016f`) | 0% (0/5) | model loops, 15 calls per scenario |
| post-fix nothink (image `dc20057e`) | **100% (5/5)** | all scenarios pass, fast |

Post-fix scenario detail (2026-05-31, image `dc20057e-cuda12`):

| # | Scenario | Result | wall | calls | notes |
|---|----------|--------|------|-------|-------|
| 1 | basic_2step | PASS | 4.11s | 2 | get_country_info → summarize |
| 2 | sequential_3step | PASS | 6.09s | 3 | fetch → analyze → report |
| 3 | error_recovery | PASS | 5.04s | 3 | recovers from TypeError |
| 4 | tool_selection | PASS | 8.57s | 4 | |
| 5 | argument_fidelity | PASS | 7.84s | 3 | |

All scenarios complete in 2–4 calls (down from 15 failed iterations).
Total bench wall time: ~32s (down from ~118s).

## Comparison: Qwen3.6 vs Gemma4 forge (both on image `dc20057e-cuda12`)

| Model | forge pass_rate | calls | notes |
|-------|-----------------|-------|-------|
| Gemma4-26B-A4B (Q4_K_M) | 20% (1/5) | 1,9,6,6,15 | basic_2step PASS (one-shot); rest fail on model behavior |
| Qwen3.6-27B (Q4_K_M) | **100% (5/5)** | 2,3,3,4,3 | all scenarios pass cleanly |

**Qwen3.6 is substantially better on forge** (100% vs 20%). The tool_result fix
was decisive for Qwen3.6 (0%→100%) and neutral for Gemma4 (20%→20%).

### Why the fix is neutral for Gemma4

Gemma4's only passing scenario (basic_2step) works via **one-shot batching**: the
model emits both required and terminal tools in a single response. The runner
executes them in order and reaches `terminal_reached` without ever needing to send
tool_results back. So the tool_result fix doesn't affect this path.

The failing Gemma4 scenarios: when the model receives tool results (now properly
contextualized by the fix), it still fails to continue correctly — it generates
text responses instead of the next tool call. This is a model behavior limitation,
not a server bug.

### Why the fix is critical for Qwen3.6

Qwen3.6 uses turn-by-turn tool calling: call one tool, receive the result, call the
next. This requires multi-turn conversation with proper tool_result context. Without
the fix, each turn was sending only role delimiters (+10 tokens) instead of actual
results, so the model kept calling the same tool in a loop (15 iterations → FAIL).
With the fix: 2–4 calls per scenario, all pass.

## Why `--think`/`--no-think` don't affect forge

The forge runner uses `forge_eval._forge.clients.anthropic.AnthropicClient`
which sends raw Anthropic SDK requests. The luce-bench `thinking_control`
injection runs in a different path (the `_prompt_thinking_control()` wrapper
in `runner.py`). For forge, neither flag appends `/think` or `/no_think` to
any message. This is intentional — the forge scenarios test tool-calling
capability under the model's default generation behavior.

## Next steps

1. ~~Rebuild image with fix~~ Done (image `dc20057e-cuda12`)
2. ~~Run forge with new image~~ Done (100% pass rate)
3. ~~Update this doc with post-fix results~~ Done
4. ~~Assess what Qwen3.6 forge ceiling looks like~~ Done — 100% (5/5)
5. ~~Consider: does the same tool_result bug affect agent_recorded?~~ No —
   agent_recorded reads text content, not tool_use blocks
6. ~~Re-run Gemma4 forge with new image~~ Done (20%→20%, neutral for Gemma4)
7. Think-mode forge for Qwen3.6: skip — --think/--no-think don't inject into
   forge runner, so result is identical
