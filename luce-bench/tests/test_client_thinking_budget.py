"""Tier-2 client-thinking-budget tests (mocked HTTP/SSE — no live server).

Covers the streaming abort + forced-`</think>` re-prompt path in
``lucebench.runner.run_case`` and the comparability/reporting rules in
``lucebench.report``:

  * reasoning_content stream over budget → abort, continuation built with
    terminator + assistant-prefill, mode="client_abort", engaged=True.
  * <think>-tag stream over budget → marking="think_tags", boundary handled.
  * unmarked stream → no abort, marking="unmarked", normal grade.
  * continuation empty / HTTP error → continuation="unsupported", no raise,
    row flagged for exclusion.
  * budget not hit → byte-identical to the no-budget path (no 2nd request).
  * answer_started_before_abort=True → flagged + excluded.
  * reporting helper: client_abort rows not pooled with single-pass; coverage
    reported.

See docs/client-thinking-budget.md.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from lucebench.report import budgeted_mode_stats
from lucebench.runner import run_case
from lucebench.schema import CanonicalRow, ClientThinking

# ────────────────────────────────────────────────────────────────────
# SSE / non-streaming mocks
# ────────────────────────────────────────────────────────────────────


def _sse_ctx(frames: list[dict | str], status: int = 200):
    """Context-manager mock yielding SSE byte lines for the given frames."""
    lines: list[bytes] = []
    for frame in frames:
        if isinstance(frame, dict):
            lines.append(("data: " + json.dumps(frame) + "\n").encode())
        else:
            lines.append(("data: " + frame + "\n").encode())
        lines.append(b"\n")
    resp = MagicMock()
    resp.__iter__ = lambda self: iter(lines)
    resp.status = status
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def _json_ctx(body: dict, status: int = 200):
    """Context-manager mock for a non-streamed JSON response (continuation)."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode()
    resp.status = status
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def _answer_response(content: str = "Answer: 42") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4},
    }


_CASE = {"id": "x", "source": "test", "kind": "integer", "question": "1+1?"}

# 200 chars of reasoning ≈ 50 tokens at char/4 — comfortably over a budget of 5.
_LONG_REASON = "x" * 200


# ────────────────────────────────────────────────────────────────────
# reasoning_content over budget → abort + continuation
# ────────────────────────────────────────────────────────────────────


def test_reasoning_content_over_budget_aborts_and_reprompts():
    """A reasoning_content stream over N aborts and fires a continuation.

    The continuation must be a SECOND request whose last message is an
    assistant-prefill carrying the captured partial reasoning + the card's
    terminator, with thinking disabled.
    """
    card = {
        "thinking_terminator_hint": "STOP-THINKING\n</think>\n\n",
        "hard_limit_reply_budget": 1234,
    }
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"reasoning_content": _LONG_REASON}}]},
        {"model": "m", "choices": [{"delta": {"reasoning_content": "more"}}]},
        "[DONE]",
    ]
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        calls.append(body)
        if body.get("stream"):
            return _sse_ctx(stream_frames)
        return _json_ctx(_answer_response("Answer: 42"))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=True,
            client_thinking_budget=5,
            model_card=card,
            model="qwen3.6-27b",
        )

    ct = row["client_thinking"]
    assert ct["mode"] == "client_abort"
    assert ct["engaged"] is True
    assert ct["marking"] == "reasoning_content"
    assert ct["continuation"] == "ok"
    assert ct["answer_started_before_abort"] is False
    assert ct["requested_budget"] == 5
    assert ct["reasoning_tokens_at_abort"] >= 5

    # Two requests: the aborted stream + the non-streamed continuation.
    assert len(calls) == 2
    cont = calls[1]
    assert cont["stream"] is False
    assert cont["max_tokens"] == 1234  # card reply reserve
    assert cont["chat_template_kwargs"] == {"enable_thinking": False}
    assert cont["reasoning_effort"] == "none"
    # Last message is the assistant prefill: partial reasoning + terminator.
    last = cont["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"].endswith("STOP-THINKING\n</think>\n\n")
    assert _LONG_REASON in last["content"]
    # The continuation answer is what's graded (lands in row content).
    assert row["content"] == "Answer: 42"


def test_continuation_uses_default_terminator_when_card_has_none():
    """No card terminator hint → fall back to </think>\\n\\n."""
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"reasoning_content": _LONG_REASON}}]},
        "[DONE]",
    ]
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        calls.append(body)
        if body.get("stream"):
            return _sse_ctx(stream_frames)
        return _json_ctx(_answer_response())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=True,
            client_thinking_budget=5,
            model_card=None,
        )
    assert row["client_thinking"]["continuation"] == "ok"
    assert calls[1]["messages"][-1]["content"].endswith("</think>\n\n")
    # Default reply reserve when no card budget.
    assert calls[1]["max_tokens"] == 4096


# ────────────────────────────────────────────────────────────────────
# <think>-tag stream over budget → think_tags marking
# ────────────────────────────────────────────────────────────────────


def test_think_tag_stream_over_budget_marks_think_tags():
    """Reasoning marked by <think> tags in content → marking=think_tags."""
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"content": "<think>" + _LONG_REASON}}]},
        {"model": "m", "choices": [{"delta": {"content": "still thinking"}}]},
        "[DONE]",
    ]
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        calls.append(body)
        if body.get("stream"):
            return _sse_ctx(stream_frames)
        return _json_ctx(_answer_response("Answer: 7"))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=True,
            client_thinking_budget=5,
        )
    ct = row["client_thinking"]
    assert ct["marking"] == "think_tags"
    assert ct["engaged"] is True
    assert ct["continuation"] == "ok"
    # The reconstructed partial reasoning (after <think>, before close) feeds
    # the assistant-prefill — it must NOT include the literal <think> tag.
    prefill = calls[1]["messages"][-1]["content"]
    assert "<think>" not in prefill
    assert _LONG_REASON in prefill


def test_think_tag_boundary_after_close_marks_answer_started():
    """Visible answer after </think> sets answer_started_before_abort.

    A long reasoning block (over budget) is followed by the close tag and a
    visible answer in one chunk — the boundary is handled and the visible
    answer flags answer_started_before_abort.
    """
    big = "y" * 200
    stream_frames = [
        {
            "model": "m",
            "choices": [{"delta": {"content": "<think>" + big + "</think>Answer: 1"}}],
        },
        "[DONE]",
    ]

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        if body.get("stream"):
            return _sse_ctx(stream_frames)
        return _json_ctx(_answer_response())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=True,
            client_thinking_budget=5,
        )
    ct = row["client_thinking"]
    assert ct["marking"] == "think_tags"
    assert ct["answer_started_before_abort"] is True


# ────────────────────────────────────────────────────────────────────
# unmarked stream → no abort
# ────────────────────────────────────────────────────────────────────


def test_unmarked_stream_never_aborts():
    """No reasoning_content and no <think> tag → unmarked, no abort, graded."""
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"content": _LONG_REASON}}]},
        {"model": "m", "choices": [{"delta": {"content": "Answer: 9"}, "finish_reason": "stop"}]},
        {"model": "m", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 60}},
        "[DONE]",
    ]
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        calls.append(body)
        return _sse_ctx(stream_frames)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=True,
            client_thinking_budget=5,
        )
    ct = row["client_thinking"]
    assert ct["marking"] == "unmarked"
    assert ct["engaged"] is False
    assert ct["mode"] == "client_abort"  # feature active, just never engaged
    assert ct["continuation"] == "skipped"
    # Exactly one request — no continuation fired for unmarked output.
    assert len(calls) == 1
    # Normal grade: the full streamed content is preserved.
    assert row["content"] == _LONG_REASON + "Answer: 9"


# ────────────────────────────────────────────────────────────────────
# continuation failure → unsupported
# ────────────────────────────────────────────────────────────────────


def test_continuation_empty_answer_is_unsupported():
    """Continuation returns an empty answer → continuation=unsupported."""
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"reasoning_content": _LONG_REASON}}]},
        "[DONE]",
    ]

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        if body.get("stream"):
            return _sse_ctx(stream_frames)
        return _json_ctx(_answer_response(""))  # empty visible answer

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=True,
            client_thinking_budget=5,
        )
    ct = row["client_thinking"]
    assert ct["continuation"] == "unsupported"
    assert ct["engaged"] is True


def test_continuation_http_error_is_unsupported_no_raise():
    """Continuation HTTP error → unsupported, no exception, truncated row kept."""
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"reasoning_content": _LONG_REASON}}]},
        "[DONE]",
    ]

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        if body.get("stream"):
            return _sse_ctx(stream_frames)
        raise ConnectionResetError("provider rejected the prefill")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=True,
            client_thinking_budget=5,
        )
    ct = row["client_thinking"]
    assert ct["continuation"] == "unsupported"
    # No exception propagated; the truncated original reasoning is preserved.
    assert row["reasoning_content"] == _LONG_REASON
    assert "error" not in row  # the abort path is not an error row


# ────────────────────────────────────────────────────────────────────
# budget not hit → byte-identical to no-budget path
# ────────────────────────────────────────────────────────────────────


def test_budget_not_hit_is_byte_identical_no_second_request():
    """Thinking under budget → no continuation, same single-request behavior."""
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"reasoning_content": "short"}}]},
        {"model": "m", "choices": [{"delta": {"content": "Answer: 2"}, "finish_reason": "stop"}]},
        {"model": "m", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        "[DONE]",
    ]
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data))
        return _sse_ctx(stream_frames)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=True,
            client_thinking_budget=1000,
        )
    assert len(calls) == 1  # no second request
    assert row["content"] == "Answer: 2"
    assert row["reasoning_content"] == "short"
    ct = row["client_thinking"]
    assert ct["engaged"] is False
    assert ct["continuation"] == "skipped"


def test_unset_budget_emits_off_block_and_one_request():
    """Unset budget → mode=off, no second request (back-compat)."""
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"reasoning_content": _LONG_REASON}}]},
        {"model": "m", "choices": [{"delta": {"content": "Answer: 2"}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data))
        return _sse_ctx(stream_frames)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(url="http://localhost:8080", case=_CASE, think=True)
    assert len(calls) == 1
    assert row["client_thinking"]["mode"] == "off"
    assert row["client_thinking"]["requested_budget"] is None


def test_budget_noop_in_nothink():
    """client_thinking_budget is a no-op in nothink — mode stays off."""
    stream_frames = [
        {"model": "m", "choices": [{"delta": {"content": _LONG_REASON}}]},
        {"model": "m", "choices": [{"delta": {"content": "Answer: 2"}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data))
        return _sse_ctx(stream_frames)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=_CASE,
            think=False,
            client_thinking_budget=5,
        )
    assert len(calls) == 1
    assert row["client_thinking"]["mode"] == "off"


# ────────────────────────────────────────────────────────────────────
# Reporting helper — comparability rules
# ────────────────────────────────────────────────────────────────────


def _row(*, mode="client_abort", continuation="ok", answer_started=False, passed=True):
    return CanonicalRow(
        case_id="c",
        graded={"pass": passed},
        client_thinking=ClientThinking(
            mode=mode,
            continuation=continuation,
            answer_started_before_abort=answer_started,
            engaged=True,
        ),
    )


def test_budgeted_stats_excludes_unsupported_and_answer_started():
    rows = [
        _row(passed=True),  # graded, pass
        _row(passed=False),  # graded, fail
        _row(continuation="unsupported", passed=True),  # excluded
        _row(answer_started=True, passed=True),  # excluded
    ]
    stats = budgeted_mode_stats(rows)
    assert stats["total"] == 4
    assert stats["graded"] == 2
    assert stats["excluded"] == 2
    assert stats["pass"] == 1
    assert stats["rate"] == 0.5
    # Coverage surfaces the shrunken denominator (2/4).
    assert stats["coverage"] == 0.5


def test_budgeted_stats_ignores_single_pass_rows():
    """client_abort rows are NOT pooled with single-pass (off-mode) rows."""
    rows = [
        _row(passed=True),  # client_abort
        CanonicalRow(case_id="s", graded={"pass": True}, client_thinking=None),  # single-pass
        CanonicalRow(
            case_id="o",
            graded={"pass": True},
            client_thinking=ClientThinking(mode="off"),
        ),  # single-pass, off
    ]
    stats = budgeted_mode_stats(rows)
    # Only the one client_abort row counts.
    assert stats["total"] == 1
    assert stats["graded"] == 1
    assert stats["coverage"] == 1.0


def test_row_stats_does_not_pool_client_abort_with_single_pass():
    """report._row_stats keeps client_abort rows out of the single-pass pool."""
    from lucebench.report import _row_stats
    from lucebench.schema import CanonicalResult

    canon = CanonicalResult(
        rows=[
            CanonicalRow(case_id="a", graded={"pass": True}, client_thinking=None),
            _row(passed=False),  # client_abort, would drag the single-pass rate down
        ]
    )
    stats = _row_stats(canon)
    # Single-pass pool sees only the 1 non-client_abort row → 100%.
    assert stats["n"] == 1
    assert stats["rate"] == 100.0
    # Budgeted block carried separately with its own coverage.
    assert stats["budgeted"]["total"] == 1
    assert stats["budgeted"]["rate"] == 0.0
