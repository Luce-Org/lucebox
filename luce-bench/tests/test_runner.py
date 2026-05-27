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
    case = {"id": "x", "source": "test", "kind": "integer", "question": "1+1?"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_chat_response(content="2"))):
        row = run_case(url="http://localhost:8080", case=case)
    assert row["case_id"] == "x"
    assert row["source"] == "test"
    assert row["content"] == "2"
    assert row["finish_reason"] == "stop"
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5
    assert row["http_status"] == 200
    assert row["wall_seconds"] is not None


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
        row = run_case(url="http://localhost:8080", case=case)
    assert row["timings"]["decode_tokens_per_sec"] == 109.4


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
