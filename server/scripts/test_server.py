import os
import struct
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from server import (
    build_app, MODEL_NAME,
    parse_tool_calls, parse_reasoning,
    normalize_stop, first_stop_match,
    _thinking_enabled, strip_closed_think_prefill, split_unclosed_thinking,
)


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]
    tokenizer.decode.return_value = "hello"
    tokenizer.apply_chat_template.return_value = "prompt"
    tokenizer.vocab_size = 151936
    return tokenizer


@pytest.fixture
def app(mock_tokenizer):
    """Build a FastAPI app with mocked daemon."""
    with patch("server.subprocess.Popen") as mock_popen:
        mock_popen.return_value.poll.return_value = None  # daemon alive
        a = build_app(
            target=Path("target.gguf"),
            draft=Path("draft.safetensors"),
            bin_path=Path("test_dflash"),
            budget=22,
            max_ctx=131072,
            tokenizer=mock_tokenizer,
            stop_ids={2},
        )
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


# ─── parse_reasoning ───────────────────────────────────────────────

class TestParseReasoning:
    def test_full_think_tags(self):
        cleaned, reasoning = parse_reasoning("<think>my reasoning</think>\n\nthe answer")
        assert cleaned == "the answer"
        assert reasoning == "my reasoning"

    def test_headless_think(self):
        """Model started in thinking — output has no <think>, just body+</think>."""
        cleaned, reasoning = parse_reasoning("chain of thought</think>\n\nvisible")
        assert cleaned == "visible"
        assert reasoning == "chain of thought"

    def test_no_think_tags_thinking_enabled(self):
        """With thinking enabled but no tags and not started_in_thinking, text is content."""
        cleaned, reasoning = parse_reasoning("all content", thinking_enabled=True)
        assert cleaned == "all content"
        assert reasoning is None

    def test_no_think_tags_thinking_disabled(self):
        """With thinking disabled, plain text passes through as content."""
        cleaned, reasoning = parse_reasoning("plain answer", thinking_enabled=False)
        assert cleaned == "plain answer"
        assert reasoning is None

    def test_started_in_thinking_no_close_tag(self):
        """Unclosed thinking still returns visible content instead of empty text."""
        cleaned, reasoning = parse_reasoning(
            "unfinished thought", started_in_thinking=True)
        assert cleaned == "unfinished thought"
        assert reasoning is None

    def test_started_in_thinking_no_close_tag_splits_final_paragraph(self):
        cleaned, reasoning = parse_reasoning(
            "I should answer exactly.\n\nlucebox-client-ok",
            started_in_thinking=True)
        assert cleaned == "lucebox-client-ok"
        assert reasoning == "I should answer exactly."

    def test_started_in_thinking_no_close_tag_splits_final_answer_marker(self):
        cleaned, reasoning = parse_reasoning(
            "Compute the sum.\n\nFinal Answer: 4",
            started_in_thinking=True)
        assert cleaned == "4"
        assert reasoning == "Compute the sum.\n\nFinal Answer:"

    def test_started_in_thinking_with_close_tag(self):
        """Full reasoning block when prompt started in thinking mode."""
        cleaned, reasoning = parse_reasoning(
            "thought body</think>the answer", started_in_thinking=True)
        assert cleaned == "the answer"
        assert reasoning == "thought body"

    def test_empty_think_block(self):
        cleaned, reasoning = parse_reasoning("<think></think>answer")
        assert cleaned == "answer"
        assert reasoning is None  # empty reasoning stripped to None

    def test_multiline_reasoning(self):
        text = "<think>line1\nline2\nline3</think>result"
        cleaned, reasoning = parse_reasoning(text)
        assert cleaned == "result"
        assert "line1" in reasoning and "line3" in reasoning

    def test_repeated_leading_think_closers_are_stripped(self):
        cleaned, reasoning = parse_reasoning("</think>\n</think>\n8")
        assert cleaned == "8"
        assert reasoning is None


class TestThinkingDefaults:
    def test_thinking_enabled_defaults_off(self):
        assert _thinking_enabled(None) is False
        assert _thinking_enabled({}) is False
        assert _thinking_enabled({"enable_thinking": True}) is True
        assert _thinking_enabled({"enable_thinking": False}) is False

    def test_strip_closed_think_prefill_at_end(self):
        assert strip_closed_think_prefill("prompt<think></think>\n") == "prompt"
        assert strip_closed_think_prefill("prompt<think>\n\n</think>\n\n") == "prompt"

    def test_strip_closed_think_prefill_only_at_end(self):
        text = "prompt<think></think>\nanswer"
        assert strip_closed_think_prefill(text) == text

    def test_split_unclosed_thinking_code_block(self):
        content, reasoning = split_unclosed_thinking(
            "Plan the function.\n\n**Final Output:** Provide the code.\n\n"
            "```python\ndef add(a, b):\n    return a + b\n```")
        assert content.startswith("```python")
        assert "return a + b" in content
        assert "Plan the function." in reasoning


# ─── parse_tool_calls ─────────────────────────────────────────────

class TestParseToolCalls:
    def test_single_tool_call(self):
        text = (
            'Sure!\n<tool_call>'
            '<function=read_file><parameter=path>test.py</parameter></function>'
            '</tool_call>'
        )
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read_file"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["path"] == "test.py"
        assert cleaned.strip() == "Sure!"

    def test_no_tool_tags(self):
        text = "Just a plain answer."
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert calls == []
        assert cleaned == text

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>'
            '<function=read_file><parameter=path>a.py</parameter></function>'
            '</tool_call>'
            '<tool_call>'
            '<function=read_file><parameter=path>b.py</parameter></function>'
            '</tool_call>'
        )
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert len(calls) == 2
        assert json.loads(calls[0]["function"]["arguments"])["path"] == "a.py"
        assert json.loads(calls[1]["function"]["arguments"])["path"] == "b.py"

    def test_multiple_parameters(self):
        text = (
            '<tool_call>'
            '<function=write_file>'
            '<parameter=path>out.txt</parameter>'
            '<parameter=content>hello world</parameter>'
            '</function></tool_call>'
        )
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["path"] == "out.txt"
        assert args["content"] == "hello world"

    def test_bare_qwen_xml_function_call(self):
        text = (
            '<function=read>\n'
            '<parameter=path>\n'
            '~/.npm-global/lib/node_modules/openclaw/skills/weather/SKILL.md\n'
            '</parameter>\n'
            '</function>\n'
            '</tool_call>'
        )
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "read", "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            }},
        }])
        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "path": "~/.npm-global/lib/node_modules/openclaw/skills/weather/SKILL.md"
        }

    def test_tool_call_id_format(self):
        text = "<tool_call><function=f><parameter=x>1</parameter></function></tool_call>"
        _, calls = parse_tool_calls(text, tools=None)
        assert calls[0]["id"].startswith("call_")
        assert calls[0]["type"] == "function"

    def test_text_before_and_after_tool_call(self):
        text = "Before\n<tool_call><function=f><parameter=x>1</parameter></function></tool_call>\nAfter"
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert len(calls) == 1
        assert "Before" in cleaned
        assert "After" in cleaned

    def test_function_signature_tool_call(self):
        text = '<function=web_search(query="Open Claw docs documentation")</function>'
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "web_search", "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            }},
        }])
        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "query": "Open Claw docs documentation"
        }

    @pytest.mark.parametrize("text", [
        '{"name":"web_search","arguments":{"query":"OpenAI docs"}}',
        '<tool_code>{"name":"web_search","arguments":{"query":"OpenAI docs"}}</tool_code>',
    ])
    def test_json_tool_call_shapes(self, text):
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "web_search", "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            }},
        }])
        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "OpenAI docs"}

    def test_multiple_mixed_tool_call_shapes(self):
        text = (
            '<function=web_search(query="OpenAI docs")</function>'
            '{"name":"read_file","arguments":{"path":"README.md"}}'
        )
        cleaned, calls = parse_tool_calls(text, tools=[
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "read_file"}},
        ])
        assert cleaned == ""
        assert [c["function"]["name"] for c in calls] == ["web_search", "read_file"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "OpenAI docs"}
        assert json.loads(calls[1]["function"]["arguments"]) == {"path": "README.md"}

    def test_unknown_tool_name_preserved_when_tools_are_known(self):
        text = '{"name":"unknown_tool","arguments":{"query":"OpenAI docs"}}'
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "web_search"},
        }])
        assert calls == []
        assert cleaned == text

    def test_malformed_function_signature_is_preserved(self):
        text = '<function=web_search(query="unterminated"</function>'
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "web_search"},
        }])
        assert calls == []
        assert cleaned == text


# ─── normalize_stop / first_stop_match ──────────────────────────────

class TestStopHelpers:
    def test_normalize_none(self):
        assert normalize_stop(None) == []

    def test_normalize_string(self):
        assert normalize_stop("stop") == ["stop"]

    def test_normalize_list(self):
        assert normalize_stop(["a", "b"]) == ["a", "b"]

    def test_normalize_empty_string(self):
        assert normalize_stop("") == []

    def test_first_stop_match_found(self):
        assert first_stop_match("hello world stop here", ["stop"]) == 12

    def test_first_stop_match_multiple(self):
        assert first_stop_match("ab cd ef", ["cd", "ab"]) == 0  # "ab" is earliest

    def test_first_stop_match_none(self):
        assert first_stop_match("hello world", ["xyz"]) == -1

    def test_first_stop_match_empty_list(self):
        assert first_stop_match("hello", []) == -1


# ─── /health endpoint ─────────────────────────────────────────────

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ─── CORS headers ─────────────────────────────────────────────────

def test_cors_headers(client):
    response = client.options(
        "/v1/models",
        headers={"Origin": "http://localhost:3000",
                 "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


# ─── GET /v1/models ────────────────────────────────────────────────

def test_models_endpoint(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == MODEL_NAME


def test_codex_models_endpoint(client):
    """Codex sends ?client_version= and expects {"models":[...]} format."""
    response = client.get("/v1/models?client_version=0.1.0")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "data" not in data  # must NOT have OpenAI format
    m = data["models"][0]
    assert m["slug"] == MODEL_NAME
    assert "context_window" in m
    assert "supported_reasoning_levels" in m
    assert m["shell_type"] == "shell_command"
    assert m["supports_parallel_tool_calls"] is False


# ─── POST /v1/chat/completions ─────────────────────────────────────

@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_non_streaming(mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = "chain of thought</think>\n\nvisible answer"
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "visible answer"
    assert data["choices"][0]["message"]["reasoning_content"] == "chain of thought"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


@patch("server.os.pipe")
@patch("server.os.read")
@patch("server.subprocess.Popen")
def test_chat_completions_uses_max_completion_tokens(
        mock_popen, mock_os_read, mock_pipe, mock_tokenizer):
    mock_popen.return_value.poll.return_value = None
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    app = build_app(
        target=Path("target.gguf"),
        draft=Path("draft.safetensors"),
        bin_path=Path("test_dflash"),
        budget=22,
        max_ctx=131072,
        tokenizer=mock_tokenizer,
        stop_ids={2},
    )
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "max_completion_tokens": 7,
        "stream": False,
    })

    assert response.status_code == 200
    writes = [
        call.args[0].decode("utf-8")
        for call in mock_popen.return_value.stdin.write.call_args_list
    ]
    command = next(write for write in writes if write.strip().endswith(" 7"))
    assert command.strip().split()[1] == "7"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_non_streaming_with_tool_call(mock_os_read, mock_pipe,
                                                        mock_tokenizer, app):
    """Non-streaming chat returns tool_calls when model outputs <tool_call>."""
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        '<tool_call>'
        '<function=read_file><parameter=path>test.py</parameter></function>'
        '</tool_call>'
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tc = data["choices"][0]["message"]["tool_calls"]
    assert len(tc) == 1
    assert tc[0]["function"]["name"] == "read_file"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_replays_raw_tool_call_text(mock_os_read, mock_pipe,
                                                     mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    raw_tool_text = (
        "Before\n"
        "<tool_call>"
        "<function=read_file><parameter=path>test.py</parameter></function>"
        "</tool_call>\n"
        "After"
    )
    mock_tokenizer.decode.side_effect = [raw_tool_text, "followup"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", -1),
        struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stream": False,
    })
    assert first.status_code == 200
    assistant_msg = first.json()["choices"][0]["message"]

    second = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "read test.py"},
            assistant_msg,
            {"role": "tool", "tool_call_id": assistant_msg["tool_calls"][0]["id"], "content": "file body"},
            {"role": "user", "content": "what next?"},
        ],
        "stream": False,
    })
    assert second.status_code == 200

    msgs = mock_tokenizer.apply_chat_template.call_args_list[-1][0][0]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"] == raw_tool_text
    assert "tool_calls" not in assistant


@patch("server.os.pipe")
@patch("server.os.read")
def test_zero_token_prompt_is_rejected_before_daemon(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.encode.return_value = []

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [],
        "stream": False,
    })

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["param"] == "messages"
    assert "zero tokens" in data["error"]["message"]
    mock_os_read.assert_not_called()


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_template_disables_thinking_by_default_and_strips_closed_prefill(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.apply_chat_template.return_value = "prompt<think></think>\n"
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })

    assert response.status_code == 200
    kwargs = mock_tokenizer.apply_chat_template.call_args_list[-1][1]
    assert kwargs["enable_thinking"] is False
    assert mock_tokenizer.encode.call_args_list[-1][0][0] == "prompt"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_template_can_explicitly_enable_thinking(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.apply_chat_template.return_value = "prompt<think>\n"
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": False,
    })

    assert response.status_code == 200
    kwargs = mock_tokenizer.apply_chat_template.call_args_list[-1][1]
    assert kwargs["enable_thinking"] is True
    assert mock_tokenizer.encode.call_args_list[-1][0][0] == "prompt<think>\n"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_template_can_enable_thinking_by_env(
        mock_os_read, mock_pipe, mock_tokenizer, app, monkeypatch):
    monkeypatch.setenv("DFLASH_ENABLE_THINKING", "1")
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.apply_chat_template.return_value = "prompt<think>\n"
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })

    assert response.status_code == 200
    kwargs = mock_tokenizer.apply_chat_template.call_args_list[-1][1]
    assert kwargs["enable_thinking"] is True
    assert mock_tokenizer.encode.call_args_list[-1][0][0] == "prompt<think>\n"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_streaming(mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.apply_chat_template.return_value = "<think>\n"
    mock_tokenizer.decode.side_effect = ["thought", "</think>", "answer"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })

    assert response.status_code == 200
    text = response.text
    assert "data: [DONE]" in text
    # Parse all SSE chunks
    chunks = [json.loads(line[6:]) for line in text.strip().split("\n\n")
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert len(chunks) >= 1
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_streaming_ignores_stray_think_closers(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.side_effect = ["</think>", "</think>", "8"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "4+4=?"}],
        "stream": True,
    })

    assert response.status_code == 200
    text = response.text
    assert "</think>" not in text
    assert '"content":"8"' in text or '"content": "8"' in text


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_streaming_bare_function_tool_call(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        "<function=read_file><parameter=path>x.py</parameter></function>"
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read x.py"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }],
        "stream": True,
    })

    assert response.status_code == 200
    chunks = [
        json.loads(line[6:])
        for line in response.text.strip().split("\n\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_delta = next(
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"][0]["delta"].get("tool_calls")
    )
    finish = next(
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c["choices"][0].get("finish_reason")
    )
    assert tool_delta["function"]["name"] == "read_file"
    assert json.loads(tool_delta["function"]["arguments"]) == {"path": "x.py"}
    assert finish == "tool_calls"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_streaming_replays_exact_raw_text_with_reasoning(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    raw_tool_turn = (
        "<think>private chain</think>"
        "visible"
        "<tool_call>"
        "<function=read_file><parameter=path>x.py</parameter></function>"
        "</tool_call>"
    )
    mock_tokenizer.decode.side_effect = [
        "<think>private chain",
        "</think>",
        "visible",
        "<tool_call><function=read_file><parameter=path>x.py</parameter></function></tool_call>",
        "followup",
    ]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", 13), struct.pack("<i", -1),
        struct.pack("<i", 14), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read x.py"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }],
        "stream": True,
    })
    assert first.status_code == 200
    chunks = [
        json.loads(line[6:])
        for line in first.text.strip().split("\n\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_delta = next(
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"][0]["delta"].get("tool_calls")
    )

    second = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "read x.py"},
            {"role": "assistant", "tool_calls": [{
                "id": tool_delta["id"],
                "type": "function",
                "function": tool_delta["function"],
            }]},
            {"role": "tool", "tool_call_id": tool_delta["id"], "content": "file body"},
            {"role": "user", "content": "what next?"},
        ],
        "stream": False,
    })
    assert second.status_code == 200

    msgs = mock_tokenizer.apply_chat_template.call_args_list[-1][0][0]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"] == raw_tool_turn
    assert "tool_calls" not in assistant


# ─── POST /v1/responses ───────────────────────────────────────────

@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_non_streaming(mock_os_read, mock_pipe, mock_tokenizer, app):
    """POST /v1/responses non-streaming returns ResponsesObject."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "hello"}],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["id"].startswith("resp_")
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"][0]["type"] == "output_text"
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["input_tokens"] + data["usage"]["output_tokens"]


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_non_streaming_string_input(mock_os_read, mock_pipe,
                                               mock_tokenizer, app):
    """Responses API accepts a plain string as input."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "hello world",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_non_streaming_started_in_thinking(mock_os_read, mock_pipe,
                                                      mock_tokenizer, app):
    """When prompt ends with <think>, reasoning without tags is not misclassified as content."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    # Simulate a chat template that prefills <think>\n
    mock_tokenizer.apply_chat_template.return_value = "prompt<think>\n"
    # Model output has no <think> tags — it's a continuation of the prefilled block
    mock_tokenizer.decode.return_value = "internal reasoning</think>\nactual answer"

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "hello"}],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    # The "actual answer" part should be the output text, not the reasoning
    assert "actual answer" in data["output_text"]
    # The reasoning should NOT leak into the output text
    assert "internal reasoning" not in data["output_text"]


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_with_instructions(mock_os_read, mock_pipe,
                                      mock_tokenizer, app):
    """Instructions are mapped to system message."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "hi",
        "instructions": "You are a coding assistant.",
    })

    assert response.status_code == 200
    # Verify apply_chat_template was called with system message
    calls = mock_tokenizer.apply_chat_template.call_args_list
    last_call = calls[-1]
    msgs = last_call[0][0]  # first positional arg
    assert msgs[0]["role"] == "system"
    assert "coding assistant" in msgs[0]["content"]


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_streaming(mock_os_read, mock_pipe, mock_tokenizer, app):
    """POST /v1/responses streaming emits proper SSE lifecycle events."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "say hello",
        "stream": True,
    })

    assert response.status_code == 200
    text = response.text
    # Must contain key SSE lifecycle events in order
    assert "event: response.created" in text
    assert "event: response.output_item.added" in text
    assert "event: response.content_part.added" in text
    assert "event: response.output_text.done" in text
    assert "event: response.content_part.done" in text
    assert "event: response.output_item.done" in text
    assert "event: response.completed" in text

    # Parse the completed event to verify structure
    for line_block in text.split("\n\n"):
        if "event: response.completed" in line_block:
            data_line = [l for l in line_block.split("\n") if l.startswith("data: ")][0]
            completed = json.loads(data_line[6:])
            assert completed["response"]["status"] == "completed"
            assert "usage" in completed["response"]
            break


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_streaming_ignores_stray_think_closers(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.side_effect = ["</think>", "</think>", "8"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "4+4=?",
        "stream": True,
    })

    assert response.status_code == 200
    text = response.text
    assert "</think>" not in text
    assert '"delta":"8"' in text or '"delta": "8"' in text


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_streaming_bare_function_tool_call(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        "<function=read_file><parameter=path>x.py</parameter></function>"
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read x.py",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }],
        "stream": True,
    })

    assert response.status_code == 200
    text = response.text
    assert "event: response.output_item.added" in text
    assert "event: response.function_call_arguments.done" in text
    assert '"type":"function_call"' in text or '"type": "function_call"' in text
    assert '"name":"read_file"' in text or '"name": "read_file"' in text
    assert '\\"path\\": \\"x.py\\"' in text


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_with_tools(mock_os_read, mock_pipe, mock_tokenizer, app):
    """POST /v1/responses with function tools maps correctly."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "read file.txt"}],
        "tools": [
            {"type": "function", "name": "read_file",
             "description": "Read a file",
             "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}
        ],
        "instructions": "You are a coding assistant.",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_object_tool_choice(mock_os_read, mock_pipe,
                                       mock_tokenizer, app):
    """POST /v1/responses with object-style tool_choice must not 422."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "read file.txt"}],
        "tools": [
            {"type": "function", "name": "read_file",
             "description": "Read a file",
             "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}
        ],
        "tool_choice": {"type": "function", "name": "read_file"},
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_function_call_output(mock_os_read, mock_pipe,
                                          mock_tokenizer, app):
    """Responses API maps function_call + function_call_output items."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [
            {"type": "message", "role": "user", "content": "read file.txt"},
            {"type": "function_call", "call_id": "call_abc123",
             "name": "read_file", "arguments": '{"path":"file.txt"}'},
            {"type": "function_call_output", "call_id": "call_abc123",
             "output": "file content here"},
        ],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"

    # Verify multi-turn message mapping: user + assistant(tool_call) + tool(output)
    calls = mock_tokenizer.apply_chat_template.call_args_list
    msgs = calls[-1][0][0]
    roles = [m["role"] for m in msgs]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_replay_raw_tool_call_text(mock_os_read, mock_pipe,
                                             mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    raw_tool_text = (
        '<tool_call>'
        '<function=read_file><parameter=path>file.txt</parameter></function>'
        '</tool_call>'
    )
    mock_tokenizer.decode.side_effect = [raw_tool_text, "followup"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", -1),
        struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "read file.txt"}],
    })
    assert first.status_code == 200
    first_output = first.json()["output"][0]

    second = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [
            {"type": "message", "role": "user", "content": "read file.txt"},
            first_output,
            {"type": "function_call_output", "call_id": first_output["call_id"], "output": "file body"},
            {"type": "message", "role": "user", "content": "what next?"},
        ],
    })
    assert second.status_code == 200

    msgs = mock_tokenizer.apply_chat_template.call_args_list[-1][0][0]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"] == raw_tool_text
    assert "tool_calls" not in assistant


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_developer_role_mapped_to_system(mock_os_read, mock_pipe,
                                                     mock_tokenizer, app):
    """Codex sends role=developer which maps to system."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [
            {"type": "message", "role": "developer",
             "content": "You are helpful."},
            {"type": "message", "role": "user", "content": "hi"},
        ],
    })

    assert response.status_code == 200
    calls = mock_tokenizer.apply_chat_template.call_args_list
    msgs = calls[-1][0][0]
    assert msgs[0]["role"] == "system"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_instructions_and_developer_merged(mock_os_read, mock_pipe,
                                                      mock_tokenizer, app):
    """Instructions + developer messages merge into one system message."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "instructions": "Top-level instructions.",
        "input": [
            {"type": "message", "role": "developer",
             "content": "Developer context."},
            {"type": "message", "role": "user", "content": "hi"},
        ],
    })

    assert response.status_code == 200
    calls = mock_tokenizer.apply_chat_template.call_args_list
    msgs = calls[-1][0][0]
    # Should be exactly one system message containing both
    system_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "Top-level instructions." in system_msgs[0]["content"]
    assert "Developer context." in system_msgs[0]["content"]


# ─── out-of-range token filtering (OverflowError regression) ───────

@patch("server.os.pipe")
@patch("server.os.read")
def test_out_of_range_token_non_streaming_returns_200(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    """Daemon emits a negative sentinel-like token (-2) that is not the EOS
    sentinel (-1).  Without filtering, tokenizer.decode([-2]) raises
    OverflowError → 500.  After the fix the token is silently dropped and
    the endpoint returns 200 with empty content rather than crashing."""
    mock_pipe.return_value = (1, 2)
    # Make decode raise for any negative token to mirror HF tokenizer behaviour
    def _decode(ids, **_kw):
        if any(t < 0 or t >= 151936 for t in ids):
            raise OverflowError("out of range integral type conversion attempted")
        return "hello"
    mock_tokenizer.decode.side_effect = _decode
    # Daemon stream: bogus token (-2) then EOS sentinel (-1)
    mock_os_read.side_effect = [struct.pack("<i", -2), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["finish_reason"] == "stop"


@patch("server.os.pipe")
@patch("server.os.read")
def test_out_of_range_token_streaming_returns_200(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    """Same contract for the streaming path: bad token is dropped, no crash."""
    mock_pipe.return_value = (1, 2)
    def _decode(ids, **_kw):
        if any(t < 0 or t >= 151936 for t in ids):
            raise OverflowError("out of range integral type conversion attempted")
        return ""
    mock_tokenizer.decode.side_effect = _decode
    mock_os_read.side_effect = [struct.pack("<i", -2), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
