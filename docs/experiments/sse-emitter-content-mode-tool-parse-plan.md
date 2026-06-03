# SSE Emitter: run `parse_tool_calls` on CONTENT-mode text (plain-text `call:<verb>{}` path)

Status: PLAN — pre-implementation. No code changes in this commit.

Branch: `fix/sse-emitter-content-mode-tool-parse`
Base: `Luce-Org/lucebox-hub:main` @ `8305b6c`
Affected files:
- `server/src/server/sse_emitter.cpp` — add CONTENT-mode finalize branch.
- `server/test/test_server_unit.cpp` — emitter-level coverage.

## 1. Problem statement (diagnosis is settled)

`SseEmitter` only transitions into `StreamMode::TOOL_BUFFER` when it detects
one of the XML-style openers (`<tool_call>`, `<function=`, `<tool_code>`)
in the streamed text (see `find_tool_start()` in
`server/src/server/sse_emitter.cpp:26-38`, gated through the
`mode_ == CONTENT` branch at line 388-423). For models like Gemma4 that
emit tool calls as plain text:

```
call:get_country_info{country: "France"}
_call:get_country_info{country_name: "France"}    # underscore artifact
```

…the emitter stays in `CONTENT` mode for the entire stream. The
final-pass `parse_tool_calls` invocation at
`server/src/server/sse_emitter.cpp:512-517` is gated on
`mode_ == TOOL_BUFFER`, so it never runs on plain-text emissions:

```cpp
if (mode_ == StreamMode::TOOL_BUFFER && !tool_buffer_.empty()) {
    auto parsed = parse_tool_calls(tool_buffer_, tools_);
    ...
}
```

The `parse_tool_calls` Pattern 5 regex (`re_call_verb_open()` at
`tool_parser.cpp:190-193`) plus the underscore-prefix sentinel
(commit `004a81b`) are correct, but unreachable in this code path.

Empirical: smoke against image `fac7e0f-cuda12` on bragi returns
`stop_reason: end_turn` for a Gemma4 response whose body contains
`_call:get_country_info{country_name: "France"}` plain text. No
`tool_use` content block is produced. This is the live regression we
are fixing.

## 2. Goal

In `SseEmitter::emit_finish()`, add a parallel finalize branch that runs
when `mode_ == CONTENT` and the accumulated content text plausibly
contains a `call:<verb>{...}` invocation. If `parse_tool_calls` returns
≥1 ToolCall, hoist them into `tool_calls_` exactly like the TOOL_BUFFER
path, strip the matched spans from accumulated content, and flip the
finish reason to `tool_calls` so the Anthropic mapping at
`http_server.cpp:2074` resolves `stop_reason="tool_use"`.

Non-goals (explicitly out of scope):
- Per-delta in-stream detection of `call:<verb>{}`. The streaming
  emitter currently sends `content` deltas as raw text as soon as the
  holdback drains; rewriting those into a `tool_use` post-hoc would
  contradict bytes already on the wire. See §6.
- Touching the existing 5 tool-call detection patterns. The
  XML/JSON/tool_code paths already work; we add a sibling branch.

## 3. Design — finalize-pass CONTENT-mode parser

### 3.1 Trigger predicate

After the existing `if (mode_ == StreamMode::TOOL_BUFFER && ...)` block
at `sse_emitter.cpp:512-617`, add a sibling branch:

```cpp
} else if (mode_ == StreamMode::CONTENT &&
           !accumulated_content_.empty() &&
           has_request_tools(tools_) &&
           looks_like_plain_text_call(accumulated_content_)) {
    auto parsed = parse_tool_calls(accumulated_content_, tools_);
    if (!parsed.tool_calls.empty()) {
        tool_calls_ = std::move(parsed.tool_calls);
        accumulated_content_ = parsed.cleaned_text;  // matched spans stripped
        fr = "tool_calls";

        // emit format-specific events for tool calls (same switch as TOOL_BUFFER)
        ...
    }
}
```

### 3.2 Cheap pre-check (`looks_like_plain_text_call`)

To avoid paying full-regex cost on every content response, gate the
parser on a tightened substring scan. Implementation:

```cpp
static bool looks_like_plain_text_call(const std::string & text) {
    // Match the tightened opener: `call:<alpha>{`. Walks the text
    // once; no heap allocation. Mirrors the sentinel logic in
    // re_call_verb_open() at a coarse granularity so we only run the
    // full std::regex pass when there's a plausible candidate.
    size_t pos = 0;
    while ((pos = text.find("call:", pos)) != std::string::npos) {
        size_t v = pos + 5;
        if (v < text.size() && (std::isalpha((unsigned char)text[v]) || text[v] == '_')) {
            // Walk verb chars; require `{` after.
            size_t w = v;
            while (w < text.size() &&
                   (std::isalnum((unsigned char)text[w]) ||
                    text[w] == '_' || text[w] == '.' ||
                    text[w] == ':' || text[w] == '-')) {
                w++;
            }
            // Allow whitespace between verb and brace (mirrors `\s*\{` in the regex).
            while (w < text.size() && std::isspace((unsigned char)text[w])) w++;
            if (w < text.size() && text[w] == '{') return true;
        }
        pos = v;
    }
    return false;
}
```

Tradeoff: the full regex pattern accepts namespaced verbs like
`call:tools.weather:get_data{...}`. The pre-check's character class
already covers `:` `.` `-`, so namespaced calls pass. Performance: O(N)
single-pass with a `find("call:")` skip, dominated by the substring
scan; no regex compile/match cost for the common no-tool-call response.

Codex review will ask whether the tightened pre-check is sufficient or
whether `std::regex_search` against a compiled-once
`call:[A-Za-z_][A-Za-z0-9_.:-]*\s*\{` regex is preferable. Decision
deferred to §5.

### 3.3 Hoist semantics — mirror the TOOL_BUFFER path

For each emitted ToolCall:
- ID: reuses the existing `generate_call_id()` from `tool_parser.cpp:31`
  via the call inside `parse_tool_calls` (`add_call` at line 450 of
  `tool_parser.cpp` already assigns `tc.id = generate_call_id();`).
  No additional ID synthesis needed in `sse_emitter.cpp`.
- `tool_memory_` remember: copy lines 518-522 of the TOOL_BUFFER branch
  (`tool_memory_->remember(ids, accumulated_raw_)`).
- Format-specific events: emit OpenAI `tool_calls` delta / Anthropic
  `content_block_start` + `input_json_delta` + `content_block_stop` /
  Responses `function_call_arguments.delta`+`.done`. Re-use the existing
  switch at lines 533-609.

### 3.4 `accumulated_content_` mutation — strip matched spans

`parse_tool_calls` already returns `cleaned_text` (the input minus all
matched spans across all 6 patterns, trimmed). Replace
`accumulated_content_` with `parsed.cleaned_text`. This is the C++
analog of `_strip_plain_text_tool_calls` at
`luce-bench/src/lucebench/areas/forge.py:144-172` — same semantics,
shared engine.

Edge: in streaming, bytes that were already sent as content deltas
remain on the wire. Stripping `accumulated_content_` only affects the
final non-streaming response shape (Anthropic message `content` array,
OpenAI `message.content`, Responses `output_text`) and any later
introspection. Streaming SSE clients see the unmodified text deltas
followed by a post-hoc tool_call event. See §6.

### 3.5 `finish_reason` bump

Local `fr` (line 511) becomes `"tool_calls"` when ≥1 ToolCall survives
the `tool_allowed` filter. The filter is enforced *inside*
`parse_tool_calls` itself — `add_call` at `tool_parser.cpp:452` has
`if (!tool_allowed(tools, fn_name)) return;` so unauthorized calls
never enter `parsed.tool_calls`. The emitter's trigger condition
`if (!parsed.tool_calls.empty())` therefore already guarantees that
`fr` only flips when at least one allow-listed call survived.

If the parser matches but everything is filtered out (all verbs
unknown to `tools_`), `fr` stays `"stop"` — matches the TOOL_BUFFER
path's `else` branch at line 610-617, which logs and keeps
`fr = "stop"`.

This is also Codex review prompt #4 (the reviewer flagged a missing
filter; the filter is actually inside `parse_tool_calls`. Documenting
here for the next reader.)

## 4. Edge cases

### 4.1 Empty / no-call content
Pre-check `looks_like_plain_text_call` returns false → bail before
regex. No measurable cost beyond a single `std::string::find("call:")`.

### 4.2 Mixed content + tool calls
`call:foo{...}` embedded in narrative prose. `parse_tool_calls`
already returns `cleaned_text` with matched spans removed and the
remainder trimmed (`tool_parser.cpp:623-647`). Verified: trailing
prose survives, only matched call spans are stripped.

### 4.3 Tool not in allowlist
`add_call` lambda inside `parse_tool_calls` rejects unauthorized verbs
via `tool_allowed(tools, fn_name)` at `tool_parser.cpp:452`. This is
the same filter the TOOL_BUFFER path relies on. No emitter-layer work
needed.

### 4.4 No tools declared (`tools_` empty)
Gate the entire new branch on `has_request_tools(tools_)` (the helper
already exists at `sse_emitter.cpp:22-24`). This mirrors the
TOOL_BUFFER trigger: `find_tool_start` is only called inside the
`has_request_tools(tools_) && ...` check at line 391. Same gating
keeps both paths consistent.

### 4.5 `<tool_call>` envelope still routes via TOOL_BUFFER
The CONTENT-mode branch only fires when `mode_ == CONTENT` at
`emit_finish` entry. If the model emitted `<tool_call>`, the emitter
transitions to TOOL_BUFFER inside `emit_token` (line 422) and that
path handles parsing. The new branch is mutually exclusive with the
old one — the `else if` ladder guarantees this. Regression risk: nil
for the existing 5 patterns.

### 4.6 Streaming
For `req.stream == true`, the per-delta SSE events already streamed
plain-text `call:foo{...}` to the client as `content` deltas BEFORE
finalize runs. Adding tool_use events at finalize produces a stream
that contains both text content AND a tool_use block — the OpenAI /
Anthropic SDKs *do* accept this shape (text + tool_use is legal), but
clients that gate on "first content type wins" will see text first.

Two options:
- (A) Apply the fix to both streaming and non-streaming, accepting that
  streaming clients see the call text in early deltas plus a tool_use
  block at the end. The accumulated_content_ field still gets cleaned
  for the final-message shape; the wire deltas are not retroactively
  rewritten.
- (B) Gate the new branch to non-streaming only (i.e., when no per-
  token deltas were emitted with content). This requires threading a
  `stream` boolean into the emitter (currently not present in
  `SseEmitter`).

Decision (will defer to Codex review prompt #2): start with option (A)
because the wire shape `text-delta + tool_use` is legal in both
Anthropic and OpenAI specs, and forge/agent_recorded uses non-streaming
exclusively. If Codex flags it as a real correctness hazard, switch to
(B) and document.

## 5. Codex review prompts (to be filled in after `lucebox codex` run)

Send Codex the plan + the following four questions verbatim:

1. **Pre-check tightness**: is the substring scan in
   `looks_like_plain_text_call` sufficient, or should we use a compiled-
   once `std::regex` (e.g.
   `call:[A-Za-z_][A-Za-z0-9_.:\-]*\s*\{`) to reduce false invocations
   of the full parser?
2. **Streaming SSE shape**: the TOOL_BUFFER finalize path emits per-tool
   SSE events (`content_block_start`/`input_json_delta`/`content_block_stop`
   in Anthropic, `tool_calls` delta in OpenAI). Should the new CONTENT-
   mode branch emit the same SSE events for streaming clients, or only
   populate the final non-streaming response fields? (Critical for
   streaming correctness — see §4.6.)
3. **Race / state-ordering**: between `parsed.cleaned_text` replacing
   `accumulated_content_` and the new SSE tool_use events being
   appended, could we ever emit a `tool_use` block whose `input`
   references text that's still present in `accumulated_content_`? Or
   would the final accumulated content contain the same call span we
   just emitted as a `tool_use`?
4. **`finish_reason` bump scope**: should `fr` flip to `tool_calls`
   only when ≥1 tool actually survives `tool_allowed` (current
   design), or whenever `parse_tool_calls` finds *any* match (even if
   filtered)? The TOOL_BUFFER path's existing behavior is the
   reference.

## Codex review

Run: 2026-05-31 against `lucebox.service` (image `fac7e0f-cuda12`,
Gemma4-26B-A4B, on bragi). Codex CLI v0.130.0 via `codex exec --model
dflash`. Verbatim answers below.

> **Q1. Is the substring-based `looks_like_plain_text_call` pre-check sufficient?**
> **Yes.** A manual substring scan for `call:` followed by a character
> check is significantly more efficient than a compiled `std::regex`
> for a `finalize()` call. Since this is an end-of-stream operation,
> the O(N) scan is negligible, and you avoid the heavy state machine
> overhead of regex for a simple pattern.

> **Q2. For streaming responses, should the new CONTENT-mode branch emit SSE tool_use events at finalize?**
> **Yes (Option A).** In streaming mode, the `content` deltas
> containing the raw `call:foo{...}` text have already been pushed to
> the client. You cannot retroactively "un-send" them. The standard
> and most compatible behavior for clients (Cline, Open WebUI) is to
> receive the raw text as prose/thought, followed by a formal
> `tool_use` block at the end.

> **Q3. Are there race / state-ordering hazards between `accumulated_content_ = parsed.cleaned_text` and the new SSE tool_use events being appended?**
> **Yes.** There is a "residue" hazard. If `parse_tool_calls` does not
> perfectly identify and strip the exact character spans of the
> `call:verb{...}` syntax from `cleaned_text`, the client will receive
> the raw tool-call string inside the final text block. This results
> in the UI displaying the tool call twice: once as literal text and
> once as a structured tool block.

> **Q4. Should `finish_reason` bump to "tool_calls" only when ≥1 tool survives `tool_allowed`, or whenever `parse_tool_calls` matches anything?**
> **Only when ≥1 tool survives `tool_allowed`.** The proposed plan
> lacks a check against the allowed tools list before setting `fr =
> "tool_calls"`. If the model emits a call that the server has
> disabled, bumping the `finish_reason` to `tool_calls` without an
> actual valid tool in the payload is a protocol violation that can
> crash or confuse SDK consumers.

> **VERDICT: REVISE.** The plan must explicitly filter
> `parsed.tool_calls` against the `tool_allowed` list before updating
> `tool_calls_` and `fr` to ensure `finish_reason` accurately reflects
> authorized tool usage.

### Response (integration / rebuttal)

- **Q1 integrated** — keep the substring pre-check as specified in §3.2.
  Codex agrees regex would be overkill at finalize. No plan change.
- **Q2 integrated** — proceed with Option A (apply fix to both streaming
  and non-streaming). The wire shape `text-delta + tool_use` is legal
  in both Anthropic and OpenAI specs per Codex. No plan change beyond
  §4.6 explicit confirmation.
- **Q3 integrated with new test** — Codex's "residue" hazard is real if
  `parse_tool_calls.cleaned_text` ever fails to strip a span. The
  parser tests already cover this for Pattern 5 (see
  `test_parse_call_verb_cleaned_text` in `test_server_unit.cpp`).
  Adding §7 emitter-level test `test_emitter_content_mode_strips_call_span_from_accumulated_text`
  to specifically guard the emitter wiring: assert that
  `em.accumulated_text()` does NOT contain the substring `call:` after
  finalize when ≥1 call was hoisted. This is a wiring regression test
  for the `accumulated_content_ = parsed.cleaned_text` line.
- **Q4 rebuttal** — Codex flags a missing `tool_allowed` filter; this
  is actually already enforced by `parse_tool_calls`'s internal
  `add_call` lambda at `tool_parser.cpp:452`
  (`if (!tool_allowed(tools, fn_name)) return;`). Calls failing the
  allowlist are dropped before they reach `result.tool_calls`. The
  plan's trigger condition is `if (!parsed.tool_calls.empty())`, so
  `fr = "tool_calls"` is set only when ≥1 allow-listed call survived.
  This matches Codex's recommendation; the perceived gap was the
  reviewer not knowing about `parse_tool_calls`'s internal filter.
  Documenting this explicitly in §3.5 for future readers.

VERDICT was REVISE; revisions applied are §7 new test
(`strips_call_span_from_accumulated_text`) and §3.5 clarification on
the existing `tool_allowed` enforcement inside `parse_tool_calls`.
Proceed to implementation.

## 6. Implementation outline (post-codex)

1. Add `looks_like_plain_text_call` static helper in
   `sse_emitter.cpp` (anonymous namespace next to `has_request_tools`).
2. Add the CONTENT-mode `else if` branch inside `emit_finish` between
   the existing TOOL_BUFFER block (line 512-617) and the format-specific
   final events switch (line 620).
3. Refactor the format-specific tool-call event emission inside the
   TOOL_BUFFER branch into a private member `emit_tool_call_events(out)`
   so both branches share the implementation. Keeps line counts down
   and avoids drift between the two paths.
4. Verify `accumulated_content_` post-mutation is consumed by the
   format-specific final events switch at line 620 (Responses uses
   `accumulated_content_` in `response.output_text.done`,
   `response.content_part.done`, and `final_output` at lines 681-712).
5. No header signature changes; the new helper is `static`.

## 7. Tests (`server/test/test_server_unit.cpp`)

Mirror the style of `test_emitter_tool_buffer_detection`,
`test_emitter_bare_function_tool_buffer_detection`, and
`test_emitter_no_tools_keeps_tool_like_text`. Add:

- `test_emitter_content_mode_plain_text_call_parsed` — feed
  `"I'll fetch it. call:get_weather{location: \"SF\"}"` to a CONTENT-mode
  emitter with `weather_tools()`. Assert: 1 ToolCall named `get_weather`
  with args `{location: "SF"}`, `accumulated_text()` no longer contains
  `call:get_weather{`, OpenAI finish_reason chunk shows `"tool_calls"`.
- `test_emitter_content_mode_no_tools_skips_plain_text_call` — same
  input, but empty tools array. Assert: no ToolCall, the call: text
  remains in `accumulated_text()`.
- `test_emitter_content_mode_underscore_prefix_call_parsed` — feed
  `"_call:get_weather{location: \"NYC\"}"`. Assert: ToolCall emitted
  (regression for the `_call:` artifact from commit `004a81b`).
- `test_emitter_content_mode_no_call_substring_skips_parser` — feed
  `"Plain prose with no tool invocations at all."`. Assert: no
  ToolCall, accumulated text unchanged, `finish_reason()` is `"stop"`.
- `test_emitter_content_mode_mixed_calls_multiple` — feed
  `"start. call:get_weather{location: \"A\"} middle. call:get_weather{location: \"B\"} end."`.
  Assert: 2 ToolCalls in order with the two locations; accumulated
  text contains `"start."`, `"middle."`, `"end."` (call spans
  stripped); no leakage of `call:`.
- `test_emitter_content_mode_malformed_call_dropped` — feed
  `"call:get_weather{unclosed"`. Assert: no ToolCall, no crash, the
  malformed text remains in `accumulated_text()` (no panic).
- `test_emitter_content_mode_does_not_double_fire_on_tool_call_xml` —
  regression guard. Feed
  `"<tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>\n</function>\n</tool_call>"`.
  Assert: exactly 1 ToolCall (TOOL_BUFFER path handled it, new branch
  did not double-emit).
- `test_emitter_content_mode_strips_call_span_from_accumulated_text` —
  Codex Q3 residue-hazard guard. Feed
  `"prefix call:get_weather{location: \"SF\"} suffix"` with weather
  tools. Assert: `em.accumulated_text().find("call:")` returns npos
  (the matched span is stripped from the visible content). Without
  this guard the emitter could double-display the call (once as
  literal text, once as a `tool_use` block).

Register each test with `RUN_TEST(...)` inside `main()` around the
existing `test_emitter_*` block (~line 3549-3560 of the file).

## 8. Don't break (regression matrix)

| Scenario | Path | Expected | Test ref |
|----------|------|----------|----------|
| `<tool_call>...` XML | TOOL_BUFFER | 1 ToolCall, no double-fire | new `does_not_double_fire` test |
| `<function=...>` bare XML | TOOL_BUFFER | 1 ToolCall | existing `test_emitter_bare_function_tool_buffer_detection` |
| `<tool_code>{json}` | TOOL_BUFFER | 1 ToolCall | existing tool_parser tests cover the parser path |
| Plain prose (no tools) | CONTENT | text preserved | existing `test_emitter_content_only_no_thinking` |
| `call:foo{...}` + tools | **new** | 1 ToolCall, text stripped | new `content_mode_plain_text_call_parsed` |
| `call:foo{...}` no tools | **new** | text preserved, no ToolCall | new `content_mode_no_tools_skips_plain_text_call` |
| Malformed `call:foo{unclosed` | **new** | text preserved, no crash | new `content_mode_malformed_call_dropped` |
| `accumulated_text()` for OpenAI Chat | both | visible text minus call spans | new `content_mode_plain_text_call_parsed` |

## 9. PR shape

- Branch: `fix/sse-emitter-content-mode-tool-parse` off `origin/main@8305b6c`.
- Two commits:
  1. `docs(experiments): plan SSE emitter CONTENT-mode tool parse` (this file).
  2. `fix(server): run parse_tool_calls on CONTENT-mode accumulated text` (impl + tests).
- Push to `easel:fix/sse-emitter-content-mode-tool-parse`.
- PR base: `Luce-Org/lucebox-hub:main`.

PR body must include:
- Diagnosis summary (verbatim from §1).
- Empirical signal: smoke against fac7e0f-cuda12 on bragi returning
  `stop_reason: end_turn` for plain-text `_call:get_country_info{...}`.
- Test count delta.
- Streaming scope decision (option A vs B from §4.6, post-Codex).
- Known limitations.

## 10. Open questions deferred to Codex

- Whether the streaming wire shape `text-delta + post-hoc tool_use`
  breaks Cline / open-webui / Anthropic SDK consumers.
- Whether `accumulated_raw_` (used for `tool_memory_->remember`)
  should be cleaned too — leaning no, since `tool_memory_` keeps the
  pre-strip raw for ID-replay matching.
