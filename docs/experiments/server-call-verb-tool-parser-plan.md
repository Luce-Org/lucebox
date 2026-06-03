# Server-side `call:<verb>{args}` tool-call parser (pattern #6)

Status: draft → codex review → implementation.
Tracking branch: `fix/server-call-verb-tool-parser` (off `origin/main`).

## Background

The 2026-05-30 gemma full bench scored forge **0/30** on RTX 3090 Ti
(`d9ecba6cc105-nvidia-geforce-rtx-3090-ti-gemma-full-2026-05-30-67f4/forge.json`).
Every row’s `iterations[0].output` shows the assistant emitting
plain-text tool invocations of the form:

- `call:get_country_info{country: "France"}`
- `call:default_api:fetch_sales_data{quarter: "Q4", year: 2024}`
- `call:execute-bead:read-file{path: "crates/foo/src/lib.rs"}`
- `call:execute-bead:list-files{path: "src/"}\n\ncall:execute-bead:read-file{path: "..."}`
- `call:execute-bead:read_file{path: "..."}`
- `call:shell{command: "rg -i auth"}`
- `call:infrastructure:get_logs{resource: \`payments-service-cluster\`, start_time: \`...\`}`
  (backtick-quoted values appear in the snapshot too)

`server/src/server/tool_parser.cpp::parse_tool_calls` has five
envelope-shaped detection patterns (XML `<tool_call>`, bare
`<function=…>`, `<function=NAME(args)>`, `<tool_code>…</tool_code>`,
bare JSON `{ "name":…, "arguments":… }`). **None** of these match the
gemma plain-text emission. As a result:

1. `SseEmitter::accumulate` buffers text, calls `parse_tool_calls`,
   gets back zero `ToolCalls`, never flips `finish_reason` to
   `tool_calls`.
2. `/v1/messages` (Anthropic) sees an empty `emitter.tool_calls()` and
   emits a `text` content block instead of a `tool_use` block
   (`server/src/server/http_server.cpp:2030–2090`).
3. forge’s WorkflowRunner expects a `ToolCall` and the row dies with
   `error_type=ValidationError`.

A client-side workaround already shipped on `feat/lucebox-docker`
(commit `deba2fd`) — see `_parse_plain_text_tool_calls` in
`luce-bench/src/lucebench/areas/forge.py`. **This PR ports that to
C++ and adds it as a sixth `parse_tool_calls` pattern**, fixing the
problem at the server.

## Goals

1. Add pattern #6 to `parse_tool_calls`: `call:<NS>?<verb>{ <relaxed-JSON-args> }`.
2. Drive the existing downstream pipeline:
   `SseEmitter::accumulate` → `finish_reason=tool_calls` → Anthropic
   `stop_reason=tool_use` with `tool_use` content blocks, or OpenAI
   `choices[].message.tool_calls`. Nothing downstream needs to change.
3. Honour the existing `tool_allowed(tools, name)` filter so callers
   that pass a constrained tool list (forge) only get back tools they
   declared.
4. Land C++ unit tests in `server/test/test_server_unit.cpp` covering
   the gemma-observed shapes plus the obvious edge cases.

## Non-goals (intentionally out of scope)

- Touching the client-side `_parse_plain_text_tool_calls` in
  `forge.py`. It stays as defense-in-depth — older deployed servers
  won’t have this fix. The PR description will note that the client
  fallback is no longer load-bearing after this merges.
- Rebuilding the docker image / running an e2e bench. The unit tests +
  a deliberately constructed `parse_tool_calls` call exercising every
  shape are sufficient for PR review.
- Touching `systemd`, the host `/home/erik/.local/bin/lucebox`
  wrapper, the running `lucebox.service` (a forge bench is in flight),
  or anything outside `server/src/server/tool_parser.cpp` +
  `server/test/test_server_unit.cpp`.

## Design

### Pattern ordering

Insert as **pattern #5** (one slot ahead of the existing bare-JSON
sweep, which becomes the new pattern #6). The four XML-envelope
patterns are lexically unambiguous and stay ahead. The reorder is
forced by a real interception hazard codex flagged in review:
`call:my_tool{"name": "inner_tool", "arguments": {}}` is a valid
gemma-shaped emission, and the bare-JSON sweep would otherwise lift
the inner `{"name": ...}` out as its own ToolCall before the new
pattern got a chance. By running the `call:…{…}` matcher first and
recording its brace-balanced span in `removals`, the bare-JSON sweep
sees that range as overlapping and skips it.

### Pattern #6 regex

```cpp
// `call:` opener: optionally preceded by start-of-string,
// whitespace, or a hard delimiter so we don't match `I'll call:foo`
// inside narrative.
static const std::regex re_call_verb_open(
    R"((^|[\s,;:\(\[\{])call:([A-Za-z0-9_.:\-]+)\s*\{)");
```

Notes:

- The leading group captures a sentinel character so we keep
  `it->position()` aligned with the start of the **prefix**, not the
  `call:` itself; the implementation computes the actual `call:`
  offset by adding the captured sentinel’s length back.
- `[A-Za-z0-9_.:\-]+` matches verbs *and* embedded namespaces in one
  capture (`execute-bead:read-file`, `default_api:fetch_sales_data`,
  bare `shell`). The verb-name passed to `add_call` strips everything
  up to and including the last `:`, so `execute-bead:read-file`
  becomes `read-file`.
- The `\s*` before `{` is intentional — the snapshot rows always emit
  `{` immediately, but tolerating optional whitespace doesn’t open new
  failure modes and matches the relaxed pattern from the Python port.
- `^` works on `std::regex` ECMAScript syntax against `text` from
  position 0; we do not rely on `multiline` mode. Subsequent matches
  rely on the explicit sentinel-character alternation, which catches
  newlines, spaces, commas, parens, square brackets, and curly braces
  — every realistic boundary in the snapshot data.

### Balanced-brace extractor

Mirror pattern #5’s scanner — track depth, skip over string literals,
**but** extend the string-literal handling to honour single quotes
and backticks (the snapshot row above has backticked values). This
diverges slightly from the Python port (which only knew `'` and `"`)
to handle the real gemma output we observed.

```cpp
// Returns one-past-close index, or std::string::npos.
static size_t balanced_braces_end(const std::string & text, size_t open) {
    int depth = 0;
    char in_str = 0;  // 0 or one of " ' `
    for (size_t i = open; i < text.size(); i++) {
        char c = text[i];
        if (in_str) {
            if (c == '\\' && i + 1 < text.size()) { i++; continue; }
            if (c == in_str) in_str = 0;
            continue;
        }
        if (c == '"' || c == '\'' || c == '`') { in_str = c; continue; }
        if (c == '{') depth++;
        else if (c == '}') {
            if (--depth == 0) return i + 1;
        }
    }
    return std::string::npos;
}
```

### Relaxed-JSON args parser

Direct port of `_coerce_relaxed_json` from `forge.py`:

1. Try `json::parse(strict)` on `"{" + body + "}"` first. Most
   strict-JSON args (`{"path": "src/"}`) succeed here.
2. On failure, rewrite the body: walk char-by-char tracking string
   state; when outside a string, look ahead for a bare-identifier
   `[A-Za-z_][A-Za-z0-9_]*` immediately followed by optional
   whitespace and `:` — wrap the identifier in double quotes. Also
   normalize single-quoted strings to double-quoted (open `'` becomes
   `"`; closing `'` becomes `"`; backtick strings get the same
   treatment). Backslash escapes are preserved.
3. Retry `json::parse` on the rewrite. If it still fails, **drop the
   single invocation** (do not throw, do not add a `ToolCall`, do not
   poison `removals`). Continue scanning past the closing brace.

The rewrite step has to leave already-quoted JSON keys alone. The
look-ahead is `(?!").+:` — i.e. only fire if we’re not already
sitting just after a `"` we already emitted. We track this via the
`out` buffer’s last char, exactly as the Python implementation does.

### Verb normalization

```cpp
std::string verb = full_match;          // e.g. "execute-bead:read-file"
size_t colon = verb.find_last_of(':');
if (colon != std::string::npos) verb = verb.substr(colon + 1);
```

`add_call(verb, args, span_start, span_end)` then passes through the
existing `tool_allowed` filter.

### Span tracking

When a successful invocation is parsed, push `{start, brace_end}` onto
`removals` so the surrounding cleanup pass strips the `call:…{…}` text
out of the assistant message (otherwise the OpenAI/Anthropic content
block would carry the literal `call:foo{…}` alongside the structured
tool call — the same double-signal hazard the Python port avoids via
`_strip_plain_text_tool_calls`).

The `start` recorded in `removals` is the index of `c` in `call:` (not
the preceding sentinel) so we don’t eat the user’s narrative space.

### Tool-allowed filter placement

Use the existing `tool_allowed(tools, verb)` helper inside
`add_call` (same path as patterns #1–5). Dropping happens **after**
regex match + JSON parse — this is cheap and consistent. Constraining
the regex to a known verb set would have been faster but it precludes
the “no tools constraint” case (forge passes a tool list; agent and
codex sometimes don’t).

## Regression safety inventory

Existing tool-parser tests in `server/test/test_server_unit.cpp`
(lines 239–317):

| Test | Body contains `call:` substring? |
|---|---|
| `test_parse_tool_call_xml` | no |
| `test_parse_bare_function_xml` | no |
| `test_parse_json_tool_call` | no |
| `test_parse_no_tools` | no |
| `test_parse_tool_code_wrapper` | no |
| `test_parse_tool_allowed_filter` | no |

Also the `emitter_*` family (lines 470–700+) uses `<tool_call>` XML
exclusively. No existing test has a `call:<verb>{` substring that
could accidentally match the new pattern. Safe.

## Test plan

Add to `server/test/test_server_unit.cpp` (the same file already
covers tool-parser unit tests; convention is one function per case
registered under the `── Tool parser ──` section). New cases:

1. `test_parse_call_verb_single` — `call:get_country_info{country: "France"}`
2. `test_parse_call_verb_back_to_back` —
   `call:get_country_info{country: "France"}call:summarize{text: "ok"}`
3. `test_parse_call_verb_namespaced` —
   `call:execute-bead:read-file{path: "crates/foo/src/lib.rs"}` → verb
   normalized to `read-file`
4. `test_parse_call_verb_snake_and_hyphen` — covers both
   `call:execute-bead:read_file{path: "..."}` and
   `call:execute-bead:list-files{path: "src/"}` with intervening
   newlines
5. `test_parse_call_verb_tool_allowed_filter` — verb `disallowed` in
   text, `tools` allows only `allowed`; result has zero `ToolCall`
6. `test_parse_call_verb_inline_prose_rejected` —
   `"Sure, I'll call:foo{x:1}"` *should* still match because the
   space before `call:` satisfies the sentinel. To exercise the
   anti-false-positive case, use `"narrative.call:foo{x:1}"` —
   no sentinel char before `call:`, regex rejects.
7. `test_parse_call_verb_malformed_args` —
   `call:foo{country: "France"` (unterminated brace) → call dropped,
   no crash, no `ToolCall`, no removal span
8. `test_parse_call_verb_inner_brace_in_string` —
   `call:foo{cmd: "echo {} ok"}` → must not break on the `{` inside
   the string; args dict has `cmd == "echo {} ok"`
9. `test_parse_call_verb_strict_json_args` —
   `call:foo{"path": "x"}` (already-quoted keys) parses on the strict
   path
10. `test_parse_call_verb_unquoted_keys` — relaxed pass kicks in:
    `call:foo{path: "x"}` → args dict has `path == "x"`
11. `test_parse_call_verb_cleaned_text` — verify the `call:…{…}`
    fragment is stripped from `result.cleaned_text` (parity with the
    XML envelope patterns).
12. `test_parse_call_verb_intercept_inner_json` (codex-requested) —
    `call:outer{"name": "inner", "arguments": {}}`. Exactly one
    `ToolCall` with `name == "outer"` and `arguments` containing the
    literal inner JSON; verifies that the reorder of patterns #5/#6
    actually defends against the bare-JSON sweep stealing the inner
    object.

These run under the existing `RUN_TEST(...)` block; add them under
the `── Tool parser ──` heading immediately after
`test_parse_tool_allowed_filter`.

### Build verification

The new code lives entirely in `server/src/server/tool_parser.cpp` —
already in `target_sources` for `test_server_unit`
(`server/CMakeLists.txt:776–788`). Build with:

```bash
cmake --build server/build --target test_server_unit && \
    server/build/test_server_unit
```

(If the lucebox-hub-285 worktree doesn’t have a build dir, configure
it the same way the repo expects — but a clean build is the operator’s
job, not part of the PR landing process.)

## Reverse-compat with the client-side fix

`luce-bench/src/lucebench/areas/forge.py::_parse_plain_text_tool_calls`
(commit `deba2fd` on `feat/lucebox-docker`) stays untouched. After
this PR merges and the docker image is rebuilt, the client-side path
becomes a no-op for any server that includes pattern #6. Leaving it
in place is harmless (the synthesized `ToolCall`s match what the
server now produces) and shields older deployed servers. The PR
description will spell this out so a reviewer doesn’t mistakenly
strip the Python fallback.

## Image rebuild + e2e (out of scope for this PR)

After merge, the operator should:

1. Rebuild the cuda12 image (`docker bake` per `docker-bake.hcl`).
2. Restart `lucebox.service` *after the current forge bench finishes*.
3. Re-run the 2026-05-30 gemma full bench and confirm forge passes
   non-trivially without the client-side `_parse_plain_text_tool_calls`
   workaround.

These steps are validation, not part of the PR diff.

## Risks / open questions

- **Multiline args.** The snapshot shows args that span multiple
  lines (`call:default_api:analyze_data{data: [{"date": ...}, ...]}`).
  The balanced-brace scanner is line-agnostic so this Just Works —
  noted here to make the implementer audit at least one multi-line
  case in the unit tests.
- **Backticked values.** `infrastructure:get_logs{resource: \`x\`}`
  uses backticks. JSON has no backtick-string concept; the relaxed
  rewrite normalizes them to double-quoted strings. This is one place
  the C++ port goes **further** than the Python port — that’s a
  conscious choice based on the real snapshot.
- **Performance.** `parse_tool_calls` is already O(n·patterns); pattern
  #6 adds a single regex sweep plus per-match work. The buffer fed
  through `SseEmitter::accumulate` is bounded by generation length;
  no realistic perf concern.

## Codex review

Captured verbatim from `codex exec` against the running lucebox
server (`http://localhost:8080/v1` / `dflash` model). The final
answer block follows; the codex run appended additional
stream-of-consciousness scratch after the answer (an artefact of the
model’s think channel leaking past the answer) — that scratch is
omitted as noise.

```
This is a high-quality implementation plan. The approach to handling
"dirty" JSON via a manual balanced-brace scanner and a
relaxation/normalization pass is a pragmatic and necessary solution
for the non-standard emissions observed in Gemma models.

However, there is one **critical architectural risk** regarding
pattern priority and one minor discrepancy in the brace-tracking
logic.

### Critical: Pattern Priority & Interception
Issue: Pattern #5 (Bare JSON sweep) is currently ordered before
Pattern #6 (the new `call:verb{...}` pattern).
Risk: If a tool call's arguments happen to contain keys named
`"name"` and `"arguments"`, Pattern #5 will intercept the substring
and incorrectly identify a tool call.
* Example: `call:my_tool{"name": "inner_tool", "arguments": {}}`
* Outcome: Pattern #5 will match the `{...}` part, creating a
  ToolCall for `inner_tool`. Because Pattern #5 is processed first,
  it will add a removal span for the inner JSON, potentially leaving
  the `call:my_tool` prefix dangling in the cleaned_text or causing
  Pattern #6 to fail/overlap.
* Recommendation: Move Pattern #6 to a higher priority (e.g., before
  Pattern #5). The `call:` prefix is a much stronger semantic signal
  than the presence of JSON keys.

### Technical Review

1. Regex & Anchors
* The sentinel list `[\s,;:\(\[\{]` is excellent. It effectively
  prevents false positives like `I'll call:foo`.
* Note on `^`: In `std::regex`, `^` matches the start of the string.
  While `std::regex` lacks a standard multiline flag, your inclusion
  of `\s` (which covers `\n`) in the sentinel list makes the pattern
  robust for tool calls starting on new lines.

2. Robustness of the Parser
* Balanced Braces: Your `balanced_braces` logic correctly handles
  string literals (including escaped quotes) and nested braces.
* Discrepancy: The plan mentions tracking `[]` (square brackets), but
  the provided implementation snippet only increments depth for `{`.
  While tracking `[]` isn't strictly required to find the outer `{}`
  of a JSON object, you should ensure it's implemented if you intend
  to follow the plan's promise, especially to handle edge cases where
  a `}` might appear inside an array (though unlikely in valid JSON).
* Relaxed JSON: The "rewrite" strategy (normalizing `'` and `` ` `` to
  `"`, and wrapping unquoted keys) is a solid way to handle the
  observed "relaxed" syntax.

3. Filter Placement
* The `tool_allowed` filter is correctly placed after the parser has
  successfully identified the verb and arguments. Constraining the
  regex itself to a known verb list would be a maintenance burden.

4. Unit Tests
* The test plan is very thorough. To make it bulletproof, I recommend
  adding these two specific cases:
    * Nested Tool Call: `call:outer{"name": "inner", "arguments": {}}`
      — specifically to verify that the priority fix (Point 1) works.
    * Complex String Content: `call:foo{cmd: "echo {not_a_brace}"}` —
      to ensure the string-aware scanner doesn't trip on braces
      inside quotes.

### Summary of Recommendations
1. Reorder patterns: Move Pattern #6 above Pattern #5.
2. Verify `[]` implementation: Ensure the brace scanner actually
   tracks `[` and `]` to match your design spec.
3. Add "Interception" test: Add a unit test where tool arguments
   mimic a JSON tool call to confirm the priority fix.
```

### Plan adjustments after review

1. **Pattern ordering — accepted.** Codex’s `call:my_tool{"name":
   "inner_tool", "arguments": {}}` example is real: gemma snapshot
   rows in the snapshot include `call:default_api:analyze_data{data:
   [{...}, ...]}` where the inner objects don’t happen to carry
   `name`/`arguments` keys today, but accepting that hazard for
   future-proofing is cheap. The new pattern moves to **#5**, demoting
   the bare-JSON sweep to **#6**. The five XML-envelope patterns stay
   ahead since they’re lexically unambiguous and pattern #6 would also
   spuriously chew their inner JSON if reordered.
2. **Square-bracket tracking — accepted with a clarification.** I
   don’t need `[`/`]` in the brace-depth counter for closing the outer
   `{`, because by the time we hit a `]` without `[` we’re already
   inside a string (handled) or the JSON is malformed (we drop the
   call). I will **still** add explicit `[`/`]` tracking to the
   scanner so a stray `}` *inside* an array (e.g. a JSON value like
   `"}"` written without quotes) doesn’t fool the close-counter. This
   matches the spirit of the plan’s prose.
3. **Interception unit test — accepted.** Add
   `test_parse_call_verb_intercept_inner_json` with body
   `call:outer{"name": "inner", "arguments": {}}`. Verify the
   resulting `ToolCall` has `name == "outer"` and `arguments ==
   {"name": "inner", "arguments": {}}` — and that there’s exactly
   **one** `ToolCall`, not two.
4. **“Complex string content” test — accepted.** Add
   `test_parse_call_verb_string_with_close_brace` with
   `call:foo{cmd: "echo {not_a_brace}"}`. Already partially covered
   by `test_parse_call_verb_inner_brace_in_string`; will reuse and
   strengthen that test rather than add a new one.

No rebuttals — codex’s feedback is all sound. Renumber: the new
pattern is **pattern #5**, the old bare-JSON sweep becomes **pattern
#6**. Update test names, header comments, and the file-top docstring
accordingly during implementation.
