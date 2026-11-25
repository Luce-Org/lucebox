"""Runner + CLI-helper tests with mocked urllib.

Unit-level coverage of the wire path:
  * `run_case` body shape — sampling fields omitted by default,
    `chat_template_kwargs` / `thinking` / `reasoning_effort` shipped
    on every request, `extra_body` merge.
  * Error handling — network failures return a `row.error` instead of
    raising; the wall_seconds field is still populated.
  * `resolve_model` — single model → returned; zero/N models → None.
  * `build_prompt` — branches per case kind.

No live server. Uses `unittest.mock.patch` on `urllib.request.urlopen`
so the tests run instantly and don't need a fixture HTTP server.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

from lucebench.cli import resolve_model
from lucebench.runner import DEFAULT_SYSTEM_PROMPT, build_prompt, run_case

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _mock_urlopen(response_body: dict, status: int = 200):
    """Return a context-manager mock that yields a fake `resp` object."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(response_body).encode()
    resp.status = status
    resp.headers = {"Content-Type": "application/json"}
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def _chat_response(content: str = "ok", usage: dict | None = None) -> dict:
    """A minimal OpenAI-shape chat-completions response."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ────────────────────────────────────────────────────────────────────
# build_prompt — case-kind dispatch
# ────────────────────────────────────────────────────────────────────


def test_build_prompt_choice_question():
    case = {
        "id": "x",
        "kind": "choice",
        "question": "Pick one",
        "choices": ["a", "b", "c"],
    }
    p = build_prompt(case)
    assert "Pick one" in p
    assert "A. a" in p and "B. b" in p and "C. c" in p
    assert "Answer: <letter>" in p


def test_build_prompt_integer_question():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    p = build_prompt(case)
    assert "1+1?" in p
    assert "Answer: <integer>" in p


def test_build_prompt_code_completion():
    case = {
        "id": "x",
        "kind": "code-completion",
        "prompt": "def add(a, b):\n    ",
    }
    p = build_prompt(case)
    assert "def add" in p
    assert "Output ONLY the function body" in p


def test_build_prompt_agent_passes_user_message_through():
    case = {
        "id": "x",
        "kind": "agent-prompt",
        "user_message": "fix bar.py",
    }
    assert build_prompt(case) == "fix bar.py"


def test_build_prompt_longctx_passes_prompt_through():
    case = {"id": "x", "kind": "longctx-frontier", "prompt": "...haystack..."}
    assert build_prompt(case) == "...haystack..."


# ────────────────────────────────────────────────────────────────────
# run_case — body shape
# ────────────────────────────────────────────────────────────────────


def _capture_body(case, **kwargs) -> dict:
    """Run a case against a mocked urlopen; return the sent body dict."""
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data)
        sent["url"] = req.full_url
        sent["headers"] = dict(req.header_items())
        return _mock_urlopen(_chat_response())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        run_case(url="http://localhost:8080", case=case, **kwargs)
    return sent


def test_run_case_omits_sampling_fields_by_default():
    """Default behavior: don't override server's card-defined sampling."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case)
    assert "temperature" not in sent["body"]
    assert "top_p" not in sent["body"]
    assert "top_k" not in sent["body"]


def test_run_case_ships_sampling_when_explicit():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, temperature=0.7, top_p=0.9, top_k=40)
    assert sent["body"]["temperature"] == 0.7
    assert sent["body"]["top_p"] == 0.9
    assert sent["body"]["top_k"] == 40


def test_run_case_top_k_zero_is_omitted():
    """top_k=0 means 'disabled' for some servers; omit so server defaults apply."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, top_k=0)
    assert "top_k" not in sent["body"]


def test_run_case_ships_extended_sampling_when_explicit():
    """min_p / presence_penalty / repetition_penalty ride when not None."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(
        case,
        min_p=0.05,
        presence_penalty=0.1,
        repetition_penalty=1.1,
    )
    assert sent["body"]["min_p"] == 0.05
    assert sent["body"]["presence_penalty"] == 0.1
    assert sent["body"]["repetition_penalty"] == 1.1


def test_run_case_extended_sampling_omitted_by_default():
    """Unset extended sampling fields are omitted from the body."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case)
    assert "min_p" not in sent["body"]
    assert "presence_penalty" not in sent["body"]
    assert "repetition_penalty" not in sent["body"]


def test_run_case_stamps_sampling_provenance_on_row():
    """sampling_source + the actual sampling dict echo onto the returned row."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_chat_response())):
        row = run_case(
            url="http://localhost:8080",
            case=case,
            stream=False,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            sampling_source="card",
        )
    assert row["sampling_source"] == "card"
    assert row["sampling"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    }


def test_run_case_thinking_control_fields_always_shipped():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    # nothink mode
    sent = _capture_body(case, think=False)
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["body"]["thinking"] == {"type": "disabled"}
    assert sent["body"]["reasoning_effort"] == "none"
    # think mode
    sent = _capture_body(case, think=True)
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert sent["body"]["thinking"] == {"type": "enabled"}
    assert sent["body"]["reasoning_effort"] == "high"


def test_run_case_reasoning_effort_flows_into_body():
    """--reasoning-effort {low,medium,high} sets the think-mode field."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    for effort in ("low", "medium", "high"):
        sent = _capture_body(case, think=True, reasoning_effort=effort)
        assert sent["body"]["reasoning_effort"] == effort
    # nothink always sends "none" regardless of the requested effort.
    sent = _capture_body(case, think=False, reasoning_effort="medium")
    assert sent["body"]["reasoning_effort"] == "none"


def test_run_case_default_reasoning_effort_is_high():
    """Back-compat: unset effort keeps the pre-flag default of 'high'."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, think=True)
    assert sent["body"]["reasoning_effort"] == "high"


def test_run_case_thinking_budget_tokens_added_in_think():
    """--thinking-budget-tokens N adds thinking.budget_tokens in think mode."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, think=True, thinking_budget_tokens=1024)
    assert sent["body"]["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_run_case_thinking_budget_tokens_noop_in_nothink():
    """budget_tokens is a no-op in nothink — the disabled block is untouched."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, think=False, thinking_budget_tokens=1024)
    assert sent["body"]["thinking"] == {"type": "disabled"}
    assert "budget_tokens" not in sent["body"]["thinking"]


def test_run_case_thinking_budget_tokens_unset_omitted():
    """Unset budget → no budget_tokens field (byte-identical to before)."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, think=True)
    assert "budget_tokens" not in sent["body"]["thinking"]


def test_run_case_card_tokens_win_over_family_map():
    """A card's thinking_control tokens drive injection over the family map.

    The card carries a distinctive token (``/deep_think``) that the
    hardcoded FAMILY_TOKENS map would never produce for a qwen id — so
    seeing it on the wire proves the card path won.
    """
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    card = {
        "name": "Qwen3.6 (custom)",
        "thinking_control": {
            "think_prompt_token": "/deep_think",
            "nothink_prompt_token": "/no_think",
            "injection_point": "user_turn_suffix",
        },
    }
    sent = _capture_body(
        case,
        think=True,
        model="qwen3.6-27b",
        model_card=card,
        thinking_control_flag="on",
    )
    assert sent["body"]["messages"][-1]["content"].endswith("/deep_think")


def test_run_case_card_provenance_stamped_on_row():
    """card_source / card_stem are echoed verbatim onto the returned row."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    row: dict = {}

    def fake_urlopen(req, timeout=None):
        return _mock_urlopen(_chat_response())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        row = run_case(
            url="http://localhost:8080",
            case=case,
            stream=False,
            card_source="bundled",
            card_stem="qwen3.6-27b",
        )
    assert row["card_source"] == "bundled"
    assert row["card_stem"] == "qwen3.6-27b"


def test_run_case_model_and_messages_shape():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, model="my-model")
    assert sent["body"]["model"] == "my-model"
    assert sent["body"]["messages"][0]["role"] == "system"
    assert sent["body"]["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert sent["body"]["messages"][1]["role"] == "user"
    assert "1+1?" in sent["body"]["messages"][1]["content"]


def test_run_case_per_case_system_prompt_wins():
    case = {
        "id": "x",
        "kind": "integer",
        "question": "1+1?",
        "system_prompt": "be terse",
    }
    sent = _capture_body(case)
    assert sent["body"]["messages"][0]["content"] == "be terse"


def test_run_case_per_case_max_tokens_overrides_arg():
    case = {"id": "x", "kind": "integer", "question": "1+1?", "max_tokens": 64}
    sent = _capture_body(case, max_tokens=16000)
    assert sent["body"]["max_tokens"] == 64


def test_run_case_extra_body_merged():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, extra_body={"my_provider_hint": "speedy"})
    assert sent["body"]["my_provider_hint"] == "speedy"


def test_run_case_auth_header_attached():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case, auth_header="Bearer sk-test")
    # urllib normalizes header keys to title-case
    auth = sent["headers"].get("Authorization") or sent["headers"].get("authorization")
    assert auth == "Bearer sk-test"


def test_run_case_hits_v1_chat_completions_endpoint():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent = _capture_body(case)
    assert sent["url"].endswith("/v1/chat/completions")


# ────────────────────────────────────────────────────────────────────
# run_case — response shape + error handling
# ────────────────────────────────────────────────────────────────────


def test_run_case_normalizes_response_into_row():
    # Non-streaming path: caller opts out, single-shot POST returns a full
    # OpenAI response dict. Streaming has its own coverage below.
    case = {"id": "x", "source": "test", "kind": "integer", "question": "1+1?"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_chat_response(content="2"))):
        row = run_case(url="http://localhost:8080", case=case, stream=False)
    assert row["case_id"] == "x"
    assert row["source"] == "test"
    assert row["content"] == "2"
    assert row["finish_reason"] == "stop"
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5
    assert row["http_status"] == 200
    assert row["wall_seconds"] is not None
    # Non-streaming rows still carry the new schema fields with None / False.
    assert row["ttft_seconds"] is None
    assert row["streaming"] is False


def test_run_case_surfaces_timings_when_server_emits_them():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    resp = _chat_response(
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "timings": {
                "prefill_ms": 12.3,
                "decode_ms": 456.7,
                "decode_tokens_per_sec": 109.4,
            },
        }
    )
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(resp)):
        row = run_case(url="http://localhost:8080", case=case, stream=False)
    assert row["timings"]["decode_tokens_per_sec"] == 109.4


# ────────────────────────────────────────────────────────────────────
# Streaming path — SSE parsing, TTFT capture, schema additions
# ────────────────────────────────────────────────────────────────────


def _mock_sse_response(frames: list[dict | str], status: int = 200):
    """Return a context-manager mock that yields an iterable of SSE byte lines.

    `frames` is a list of either:
      * dict → wrapped as `data: <json>\\n\\n` and split into 2 lines
        (data line + blank terminator)
      * str  → wrapped as `data: <str>\\n\\n` verbatim (use for [DONE]
        or malformed payloads)
    """
    lines: list[bytes] = []
    for frame in frames:
        if isinstance(frame, dict):
            lines.append(("data: " + json.dumps(frame) + "\n").encode())
        else:
            lines.append(("data: " + frame + "\n").encode())
        lines.append(b"\n")  # blank line = frame terminator

    resp = MagicMock()
    resp.__iter__ = lambda self: iter(lines)
    resp.status = status
    resp.headers = {"Content-Type": "text/event-stream"}
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def test_run_case_streaming_captures_ttft_and_concatenates_content():
    """A streamed response with 3 content deltas should:
      * stamp ttft on the first delta with non-empty content
      * concatenate all deltas into row["content"]
      * pick up usage from the final chunk
      * mark streaming=True
    """
    case = {"id": "x", "source": "test", "kind": "integer", "question": "1+1?"}
    frames = [
        # Role-only opener — must NOT stamp ttft (no real text yet).
        {
            "model": "mock",
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
        },
        # First text delta — ttft stamps here.
        {"model": "mock", "choices": [{"delta": {"content": "The "}, "finish_reason": None}]},
        {"model": "mock", "choices": [{"delta": {"content": "answer is "}, "finish_reason": None}]},
        {"model": "mock", "choices": [{"delta": {"content": "42."}, "finish_reason": "stop"}]},
        # Usage-only final chunk (OpenAI shape with include_usage=true).
        {
            "model": "mock",
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "timings": {
                    "prefill_ms": 45.0,
                    "decode_ms": 120.0,
                    "decode_tokens_per_sec": 58.3,
                },
            },
        },
        "[DONE]",
    ]
    with patch("urllib.request.urlopen", return_value=_mock_sse_response(frames)):
        row = run_case(url="http://localhost:8080", case=case)
    assert row["streaming"] is True
    assert row["content"] == "The answer is 42."
    assert row["finish_reason"] == "stop"
    assert row["prompt_tokens"] == 12
    assert row["completion_tokens"] == 7
    assert row["timings"]["decode_tokens_per_sec"] == 58.3
    # TTFT must be set (non-None) and bounded — wall_seconds is an upper bound.
    assert isinstance(row["ttft_seconds"], float)
    assert row["ttft_seconds"] >= 0
    assert row["ttft_seconds"] <= row["wall_seconds"] + 0.01


def test_run_case_streaming_request_sets_stream_true_and_include_usage():
    """The wire body must carry stream=true and stream_options.include_usage."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data)
        sent["headers"] = dict(req.header_items())
        return _mock_sse_response(
            [
                {"model": "m", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        run_case(url="http://localhost:8080", case=case)
    assert sent["body"]["stream"] is True
    assert sent["body"]["stream_options"] == {"include_usage": True}
    # Accept: text/event-stream is what gateways inspect for SSE routing.
    accept = sent["headers"].get("Accept") or sent["headers"].get("accept")
    assert accept == "text/event-stream"


def test_run_case_streaming_captures_reasoning_content_for_ttft():
    """A reasoning-content delta (no visible content yet) should still
    stamp TTFT — that's "first useful token" from the user's POV."""
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    frames = [
        # Role-only opener.
        {"model": "m", "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        # Reasoning-only chunk — TTFT must stamp here.
        {
            "model": "m",
            "choices": [{"delta": {"reasoning_content": "thinking..."}, "finish_reason": None}],
        },
        # Visible content arrives later.
        {"model": "m", "choices": [{"delta": {"content": "2"}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    with patch("urllib.request.urlopen", return_value=_mock_sse_response(frames)):
        row = run_case(url="http://localhost:8080", case=case)
    assert row["reasoning_content"] == "thinking..."
    assert row["content"] == "2"
    assert row["ttft_seconds"] is not None


def test_run_case_stream_false_omits_include_usage_and_uses_application_json():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data)
        sent["headers"] = dict(req.header_items())
        return _mock_urlopen(_chat_response())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        run_case(url="http://localhost:8080", case=case, stream=False)
    assert sent["body"]["stream"] is False
    assert "stream_options" not in sent["body"]
    accept = sent["headers"].get("Accept") or sent["headers"].get("accept")
    # Non-streaming runs don't override Accept — urllib defaults to */*.
    assert accept != "text/event-stream"


def test_run_case_returns_error_row_on_network_failure():
    """Network failures must NOT raise — they return an error row."""
    case = {"id": "x", "source": "test", "kind": "integer", "question": "1+1?"}
    with patch(
        "urllib.request.urlopen",
        side_effect=ConnectionRefusedError("[Errno 111] Connection refused"),
    ):
        row = run_case(url="http://localhost:9999", case=case)
    assert row["pass"] is False
    assert "ConnectionRefusedError" in (row["error"] or "")
    assert row["http_status"] is None
    assert row["wall_seconds"] is not None


def test_run_case_returns_error_row_on_timeout():
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        row = run_case(url="http://localhost:8080", case=case, timeout_s=1)
    assert row["pass"] is False
    assert "TimeoutError" in (row["error"] or "")


# ────────────────────────────────────────────────────────────────────
# resolve_model — /v1/models autopick
# ────────────────────────────────────────────────────────────────────


def test_resolve_model_picks_single_model():
    payload = {"data": [{"id": "qwen3.6", "object": "model"}], "object": "list"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert resolve_model("http://localhost:8080") == "qwen3.6"


def test_resolve_model_first_for_short_list():
    """Servers exposing a small list (<5) auto-resolve to the first model.
    The caller surfaces the full list so the choice is visible."""
    payload = {"data": [{"id": "a"}, {"id": "b"}], "object": "list"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert resolve_model("http://localhost:8080") == "a"


def test_resolve_model_returns_none_for_long_list():
    """Gateways exposing 5+ models do NOT auto-resolve — explicit --model required."""
    payload = {"data": [{"id": f"m{i}"} for i in range(6)], "object": "list"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert resolve_model("http://localhost:8080") is None


def test_resolve_model_returns_none_for_zero():
    with patch(
        "urllib.request.urlopen", return_value=_mock_urlopen({"data": [], "object": "list"})
    ):
        assert resolve_model("http://localhost:8080") is None


def test_resolve_model_returns_none_when_endpoint_missing():
    """Servers that don't speak OpenAI /v1/models → None, no exception."""
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError("http://x", 404, "Not Found", {}, io.BytesIO(b"")),
    ):
        assert resolve_model("http://localhost:8080") is None


def test_resolve_model_returns_none_on_connection_error():
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
        assert resolve_model("http://localhost:8080") is None


def test_resolve_model_returns_none_for_invalid_json():
    """Server returns 200 but not JSON → None, no exception."""
    resp = MagicMock()
    resp.read.return_value = b"<html>not json</html>"
    resp.status = 200
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=ctx):
        assert resolve_model("http://localhost:8080") is None


def test_resolve_model_attaches_auth_header_when_provided():
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        return _mock_urlopen({"data": [{"id": "x"}], "object": "list"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resolve_model("http://localhost:8080", auth_header="Bearer sk-or-test")
    auth = captured["headers"].get("Authorization") or captured["headers"].get("authorization")
    assert auth == "Bearer sk-or-test"
