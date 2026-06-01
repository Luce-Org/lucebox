"""Tests for the forge area's plain-text tool-call synthesis.

Background — the 2026-05-30 gemma full bench scored forge 0/30 cases
with ``error_type = "ValidationError"`` on every row. The root cause
was twofold:

1. The recording client was calling ``TextResponse(text=...)`` but the
   forge ``TextResponse`` field is named ``content`` — every call
   raised a pydantic ``ValidationError`` which became the per-row
   ``error_type``.
2. Even with #1 fixed, gemma emits ``call:<verb>{args}`` as plain text
   in the response (not as Anthropic ``tool_use`` content blocks). The
   old client surfaced text-only responses, so forge never saw a
   single tool call and would have nudged forever.

The fix synthesizes ``ToolCall`` entries from the plain-text
emissions client-side. These tests cover the parsing/synthesis helper
in isolation — no live server, no anthropic SDK round-trip.
"""

from __future__ import annotations

from lucebench.areas.forge import (
    _coerce_relaxed_json,
    _parse_plain_text_tool_calls,
    _strip_plain_text_tool_calls,
)

# ───────────────────────── _parse_plain_text_tool_calls ─────────────────────────


def test_no_call_pattern_returns_empty():
    """Plain prose with no ``call:<verb>{`` token → no synthesized calls."""
    assert _parse_plain_text_tool_calls("Hello, world.") == []
    assert _parse_plain_text_tool_calls("") == []


def test_single_call_relaxed_json():
    """Single ``call:foo{bar: "baz"}`` → one call, name=foo, input={bar:"baz"}."""
    calls = _parse_plain_text_tool_calls('call:foo{bar: "baz"}')
    assert len(calls) == 1
    assert calls[0] == {"name": "foo", "input": {"bar": "baz"}}


def test_back_to_back_calls_preserve_order():
    """The real 2026-05-30 gemma shape: two concatenated calls + trailing prose."""
    text = (
        'call:get_country_info{country: "France"}'
        'call:summarize{text: "The capital of France is Paris."}'
        " trailing prose that should be ignored"
    )
    calls = _parse_plain_text_tool_calls(text)
    assert len(calls) == 2
    assert calls[0] == {
        "name": "get_country_info",
        "input": {"country": "France"},
    }
    assert calls[1] == {
        "name": "summarize",
        "input": {"text": "The capital of France is Paris."},
    }


def test_snake_case_and_hyphenated_verbs():
    """Snake_case (``get_country_info``) and kebab-case (``read-file``) names both parse."""
    text = (
        'call:get_country_info{country: "France"}'
        'call:read-file{path: "foo.py"}'
    )
    calls = _parse_plain_text_tool_calls(text)
    assert [c["name"] for c in calls] == ["get_country_info", "read-file"]


def test_namespaced_verb():
    """``call:namespace:verb{...}`` (codex-mini / DDX shape) parses verb-with-ns intact."""
    calls = _parse_plain_text_tool_calls('call:shell:run{cmd: "ls"}')
    assert len(calls) == 1
    assert calls[0]["name"] == "shell:run"
    assert calls[0]["input"] == {"cmd": "ls"}


def test_strict_json_args_also_accepted():
    """If the model happens to emit valid strict JSON, that's still fine."""
    calls = _parse_plain_text_tool_calls('call:foo{"bar": "baz", "n": 7}')
    assert len(calls) == 1
    assert calls[0]["input"] == {"bar": "baz", "n": 7}


def test_malformed_args_skipped_no_crash():
    """An unparseable args block is dropped; surrounding calls survive."""
    text = (
        'call:good{x: "ok"}'
        # Intentionally broken — bare value `??!` is neither a quoted
        # string nor a valid identifier-keyed value.
        'call:bad{!!malformed!!}'
        'call:also_good{y: "ok"}'
    )
    calls = _parse_plain_text_tool_calls(text)
    # The bad invocation is dropped; the two good ones flank it.
    names = [c["name"] for c in calls]
    assert "good" in names
    assert "also_good" in names
    assert "bad" not in names


def test_unbalanced_braces_terminates_scan():
    """Unbalanced ``{`` stops scanning — we don't synthesize a tool from a half-call."""
    # Opens but never closes — parser bails.
    calls = _parse_plain_text_tool_calls('call:foo{country: "France"')
    assert calls == []


def test_nested_braces_handled():
    """Nested ``{}`` inside the args (e.g. an object value) doesn't confuse the matcher."""
    text = 'call:update_plan{plan: {step: "one", done: true}}'
    calls = _parse_plain_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "update_plan"
    assert calls[0]["input"] == {"plan": {"step": "one", "done": True}}


def test_string_with_brace_inside_does_not_break_matching():
    """A ``}`` inside a quoted string doesn't close the args block prematurely."""
    text = 'call:say{text: "look: } a brace"}'
    calls = _parse_plain_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["input"] == {"text": "look: } a brace"}


# ─────────────────────── _coerce_relaxed_json edge cases ────────────────────────


def test_relaxed_json_strict_path():
    """Strict JSON parses through the first branch without rewriting."""
    assert _coerce_relaxed_json('{"a": 1}') == {"a": 1}


def test_relaxed_json_bare_keys_rewritten():
    """Unquoted keys get quoted before re-parsing."""
    assert _coerce_relaxed_json('{a: 1, b: "two"}') == {"a": 1, "b": "two"}


def test_relaxed_json_nested():
    """Bare-key rewriting recurses into nested objects."""
    assert _coerce_relaxed_json(
        '{outer: {inner: "v"}}',
    ) == {"outer": {"inner": "v"}}


# ──────────────────────── _strip_plain_text_tool_calls ─────────────────────────


def test_strip_removes_call_spans():
    """The stripper removes ``call:<verb>{...}`` spans from reasoning text."""
    text = 'Trying:\ncall:foo{a: "b"}\ncall:bar{c: "d"}\nDone.'
    assert _strip_plain_text_tool_calls(text) == "Trying:\n\n\nDone."


def test_strip_preserves_text_when_no_calls():
    """No-op on plain prose."""
    assert _strip_plain_text_tool_calls("just prose") == "just prose"


def test_strip_handles_empty_input():
    """Empty/None safe."""
    assert _strip_plain_text_tool_calls("") == ""


def test_underscore_prefix_call_verb():
    """Regression for 2026-05-31 gemma smoke test against lucebox-hub@8039911:
    the server now sometimes emits a leading underscore — `_call:foo{...}` —
    as a SentencePiece tokenizer artifact post-channel-routing. Without
    handling the `_` prefix, the parser misses every such invocation.
    """
    text = '_call:get_country_info{country: "France"}\n\nresponse.'
    calls = _parse_plain_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_country_info"
    assert calls[0]["input"] == {"country": "France"}


def test_underscore_prefix_among_back_to_back_calls():
    """Multiple invocations, some preceded by `_`, all should match."""
    text = '_call:foo{x:1}call:bar{y:2}_call:baz{z:3}'
    calls = _parse_plain_text_tool_calls(text)
    names = [c["name"] for c in calls]
    assert names == ["foo", "bar", "baz"]
