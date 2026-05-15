import os
import struct
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from server import (
    build_app, MODEL_NAME,
    parse_tool_calls, parse_reasoning,
    normalize_stop, first_stop_match,
    PROPS_SCHEMA, SERVER_VERSION,
    _API_ENDPOINTS, _capabilities, _effective_kv_type,
    _resolve_server_version, _runtime_backend,
)


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]
    tokenizer.decode.return_value = "hello"
    tokenizer.apply_chat_template.return_value = "prompt"
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
        """Truncated reasoning when prompt started in thinking mode."""
        cleaned, reasoning = parse_reasoning(
            "unfinished thought", started_in_thinking=True)
        assert cleaned == ""
        assert reasoning == "unfinished thought"

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


# ─── GET /props ────────────────────────────────────────────────────

_PROPS_TOP_KEYS = {
    "default_generation_settings", "model_alias", "model_path", "build_info",
    "speculative_mode", "server", "model", "runtime", "reasoning",
    "speculative", "sampling", "pflash", "prefix_cache", "full_cache",
    "tool_replay", "daemon", "api",
}


def _build_props_app(mock_tokenizer, *, arch="qwen35", draft=Path("d.safetensors"),
                     prefill_cfg=None, extra_daemon_args=None,
                     prefix_cache_slots=4, prefill_cache_slots=4):
    """Build an app with mocked daemon, tuned for /props tests."""
    mock_tokenizer.name_or_path = "Qwen/Qwen3.5-27B"
    with patch("server.subprocess.Popen") as mock_popen:
        mock_popen.return_value.poll.return_value = None
        return build_app(
            target=Path("target.gguf"),
            draft=draft,
            bin_path=Path("test_dflash"),
            budget=22,
            max_ctx=131072,
            tokenizer=mock_tokenizer,
            stop_ids={2},
            prefill_cfg=prefill_cfg,
            drafter_tokenizer=mock_tokenizer if prefill_cfg else None,
            prefix_cache_slots=prefix_cache_slots,
            prefill_cache_slots=prefill_cache_slots,
            arch=arch,
            extra_daemon_args=extra_daemon_args,
        )


def test_props_endpoint_shape(client):
    response = client.get("/props")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == _PROPS_TOP_KEYS
    assert body["server"]["props_schema"] == PROPS_SCHEMA
    assert body["server"]["version"] == SERVER_VERSION
    assert body["server"]["name"] == "luce-dflash"
    assert body["model_alias"] == MODEL_NAME
    assert body["model_path"] == "target.gguf"
    assert body["build_info"] == f"luce-dflash v{SERVER_VERSION} props_schema={PROPS_SCHEMA}"


def test_props_llama_compat_fields(client):
    body = client.get("/props").json()
    assert body["default_generation_settings"] == {
        "n_ctx": 131072,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    }
    assert body["runtime"]["backend"]
    assert body["speculative_mode"] == "dflash"
    assert body["reasoning"] == {
        "supported": True,
        "default": None,
        "supported_efforts": ["medium"],
    }
    assert body["sampling"] == {
        "capabilities": {
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_top_k": True,
            "supports_frequency_penalty": True,
            "supports_seed": True,
        },
    }
    assert "max_ctx" not in body["runtime"]
    assert "id" not in body["model"]
    assert "target_path" not in body["model"]
    assert all(not key.startswith("supports_") for key in body["sampling"])
    assert "default_enabled" not in body["reasoning"]


def test_props_version_reads_pyproject():
    # Best effort: when pyproject is reachable the value is the [project]
    # version. We don't assert a literal because that drifts with releases.
    version = _resolve_server_version()
    assert isinstance(version, str)
    assert version
    if version != "0.0.0+unknown":
        # pyproject was readable — version should look like semver-ish.
        assert version[0].isdigit()


def test_props_version_falls_back_when_pyproject_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("server._PYPROJECT", tmp_path / "does_not_exist.toml")
    assert _resolve_server_version() == "0.0.0+unknown"


def test_runtime_backend_prefers_env(monkeypatch):
    monkeypatch.setenv("DFLASH_RUNTIME_BACKEND", "HIP")
    assert _runtime_backend(Path("missing")) == "hip"


def test_runtime_backend_reads_cmake_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("DFLASH_RUNTIME_BACKEND", raising=False)
    monkeypatch.delenv("DFLASH27B_GPU_BACKEND", raising=False)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        "DFLASH27B_GPU_BACKEND:STRING=hip\n",
        encoding="utf-8",
    )
    assert _runtime_backend(build_dir / "test_dflash") == "hip"


def test_props_arch_qwen35(client):
    body = client.get("/props").json()
    assert body["model"]["arch"] == "qwen35"
    assert body["model"]["draft_path"] is not None
    assert body["reasoning"]["supported"] is True
    assert body["speculative"]["enabled"] is True
    assert body["speculative"]["ddtree_budget"] == 22


def test_props_arch_laguna(mock_tokenizer):
    app = _build_props_app(mock_tokenizer, arch="laguna", draft=None)
    body = TestClient(app).get("/props").json()
    assert body["model"]["arch"] == "laguna"
    assert body["model"]["draft_path"] is None
    assert body["reasoning"] == {
        "supported": False,
        "default": None,
        "supported_efforts": [],
    }
    assert body["speculative"]["enabled"] is False
    assert body["speculative"]["ddtree_budget"] is None
    assert body["speculative_mode"] == "off"


def test_props_pflash_disabled(client):
    body = client.get("/props").json()
    p = body["pflash"]
    assert p["enabled"] is False
    assert p["mode"] == "off"
    for k in ("threshold", "keep_ratio", "drafter_gguf",
              "bsa_enabled", "bsa_alpha", "lm_head_fix"):
        assert p[k] is None, f"expected null pflash.{k} when disabled"


def test_props_pflash_enabled(mock_tokenizer, monkeypatch):
    from _prefill_hook import PrefillConfig
    cfg = PrefillConfig(
        mode="auto",
        threshold=32000,
        keep_ratio=0.05,
        drafter_gguf=Path("/tmp/drafter.gguf"),
        drafter_tokenizer_id="Qwen/Qwen3-0.6B",
        skip_park=False,
    )
    monkeypatch.setenv("DFLASH_FP_USE_BSA", "1")
    monkeypatch.setenv("DFLASH_FP_ALPHA", "0.85")
    app = _build_props_app(mock_tokenizer, prefill_cfg=cfg)
    body = TestClient(app).get("/props").json()
    p = body["pflash"]
    assert p["enabled"] is True
    assert p["mode"] == "auto"
    assert p["threshold"] == 32000
    assert p["keep_ratio"] == pytest.approx(0.05)
    assert p["drafter_gguf"] == "/tmp/drafter.gguf"
    assert p["bsa_enabled"] is True
    assert p["bsa_alpha"] == pytest.approx(0.85)
    assert body["speculative_mode"] == "pflash"


def test_props_target_sharding_disables_caches(mock_tokenizer):
    app = _build_props_app(
        mock_tokenizer,
        extra_daemon_args=["--target-gpus=0,1"],
    )
    body = TestClient(app).get("/props").json()
    assert body["runtime"]["target_sharding"] is True
    assert body["prefix_cache"]["capacity"] == 0
    assert body["full_cache"]["enabled"] is False


def test_props_target_sharding_false_on_laguna_even_when_args_passed(mock_tokenizer):
    """Laguna's daemon-spawn path discards extra_daemon_args (see server.py
    ~line 788 — only the qwen35 branch calls cmd.extend). /props must not
    claim sharding is active when the flag was silently dropped."""
    app = _build_props_app(
        mock_tokenizer,
        arch="laguna",
        draft=None,
        extra_daemon_args=["--target-gpus=0,1"],
    )
    body = TestClient(app).get("/props").json()
    assert body["runtime"]["target_sharding"] is False


def test_props_endpoints_match_app_routes(client, app):
    declared = set(client.get("/props").json()["api"]["endpoints"])
    actual: set[str] = set()
    for r in app.routes:
        if not isinstance(r, APIRoute):
            continue
        for m in r.methods:
            if m in {"GET", "POST"}:
                actual.add(f"{m} {r.path}")
    # FastAPI auto-routes we don't care to advertise.
    actual -= {
        "GET /openapi.json", "GET /docs", "GET /redoc",
        "GET /docs/oauth2-redirect",
    }
    assert declared == actual


def test_capabilities_arch_gated():
    assert _capabilities("qwen35") == {
        "reasoning_supported": True,
        "speculative_supported": True,
        "tools_supported": True,
    }
    assert _capabilities("laguna") == {
        "reasoning_supported": False,
        "speculative_supported": False,
        "tools_supported": False,
    }


def test_effective_kv_type_qwen35_defaults(monkeypatch):
    for env in ("DFLASH27B_KV_F16", "DFLASH27B_KV_Q4", "DFLASH27B_KV_TQ3",
                "DFLASH27B_KV_K", "DFLASH27B_KV_V"):
        monkeypatch.delenv(env, raising=False)
    # qwen35 default per dflash/src/kv_quant.cpp:160 is q4_0 for both axes.
    assert _effective_kv_type("k", "qwen35") == "q4_0"
    assert _effective_kv_type("v", "qwen35") == "q4_0"


def test_effective_kv_type_qwen35_per_axis_override(monkeypatch):
    monkeypatch.setenv("DFLASH27B_KV_TQ3", "1")
    monkeypatch.setenv("DFLASH27B_KV_V", "q4_0")
    monkeypatch.delenv("DFLASH27B_KV_K", raising=False)
    assert _effective_kv_type("k", "qwen35") == "tq3_0"
    assert _effective_kv_type("v", "qwen35") == "q4_0"


def test_effective_kv_type_laguna_ignores_legacy_and_v(monkeypatch):
    # Laguna only reads DFLASH27B_KV_K and applies it to both axes.
    monkeypatch.setenv("DFLASH27B_KV_TQ3", "1")
    monkeypatch.setenv("DFLASH27B_KV_V", "q4_0")
    monkeypatch.delenv("DFLASH27B_KV_K", raising=False)
    assert _effective_kv_type("k", "laguna") == "q8_0"
    assert _effective_kv_type("v", "laguna") == "q8_0"
    monkeypatch.setenv("DFLASH27B_KV_K", "q4_0")
    assert _effective_kv_type("k", "laguna") == "q4_0"
    assert _effective_kv_type("v", "laguna") == "q4_0"


def test_prefix_cache_stats_disabled():
    from prefix_cache import PrefixCache
    pc = PrefixCache.__new__(PrefixCache)
    pc.disabled = True
    pc.cap = 0
    assert pc.stats() == {"capacity": 0, "in_use": 0, "lifetime_hits": 0}


def test_prefix_cache_lifetime_hits_increments(mock_tokenizer):
    """Driving lookup() across an eviction must still see the lifetime
    counter — it is NOT a sum over surviving entries."""
    from prefix_cache import PrefixCache, hash_prefix
    pc = PrefixCache(
        daemon_stdin=MagicMock(),
        await_reply=lambda *_a, **_k: None,
        daemon_lock=MagicMock(),
        tokenizer=mock_tokenizer,
        kv_k_type="q8_0",
        fa_window=2048,
        cap=2,
    )
    # Force-disable boundary detection so we can register synthetic entries.
    pc.markers = {"family": "manual", "sys_role_prefix": (),
                  "end_msg_seqs": [], "next_role_starts": []}
    # Inject two entries directly and hit each twice.
    ids_a = [1, 2, 3, 4]
    ids_b = [5, 6, 7, 8]
    key_a = hash_prefix(ids_a, pc.kv_k_type, pc.fa_window)
    key_b = hash_prefix(ids_b, pc.kv_k_type, pc.fa_window)
    pc.entries[key_a] = 0
    pc.entries[key_b] = 1
    # Stub out the boundary detector to return both candidate cuts.
    pc._all_boundaries = lambda ids: [len(ids)]
    pc.lookup(ids_a)
    pc.lookup(ids_a)
    pc.lookup(ids_b)
    assert pc.stats()["lifetime_hits"] == 3
    # Evict everything; counter persists.
    pc.entries.clear()
    assert pc.stats()["in_use"] == 0
    assert pc.stats()["lifetime_hits"] == 3


def test_full_cache_disk_bytes_snapshot_updates_on_mutation(mock_tokenizer, tmp_path):
    """confirm_full_snap and _retire_full_entry must refresh the snapshot
    so /props never has to walk the filesystem."""
    from prefix_cache import PrefixCache
    pc = PrefixCache(
        daemon_stdin=MagicMock(),
        await_reply=lambda *_a, **_k: None,
        daemon_lock=MagicMock(),
        tokenizer=mock_tokenizer,
        kv_k_type="q8_0",
        fa_window=2048,
        cap=1,
    )
    pc.markers = {"family": "manual", "sys_role_prefix": (),
                  "end_msg_seqs": [], "next_role_starts": []}
    pc.init_full_cache(full_cap=2, cache_dir=str(tmp_path))
    assert pc.full_stats()["disk_bytes"] == 0

    # Source bin file the daemon would have written.
    src = tmp_path / "cur.bin"
    src.write_bytes(b"\x01\x00\x00\x00" * 16)

    ids = [10, 20, 30]
    slot, _ = pc.prepare_full_snap(ids)
    pc.confirm_full_snap(slot, ids, src, cur_ids_len=16)
    snap = pc.full_stats()
    assert snap["enabled"] is True
    assert snap["in_use"] == 1
    assert snap["disk_bytes"] > 0
    after_add = snap["disk_bytes"]

    # Direct retire of the entry should refresh the snapshot to 0.
    (key, entry), = pc.full_entries.items()
    pc._retire_full_entry(key, entry, remove_files=True)
    after_evict = pc.full_stats()
    assert after_evict["in_use"] == 0
    assert after_evict["disk_bytes"] == 0
    assert after_evict["disk_bytes"] < after_add


def test_tool_memory_stats():
    from tool_memory import ToolMemory
    tm = ToolMemory(max_entries=100, max_bytes=4096)
    s = tm.stats()
    assert s == {
        "max_entries": 100,
        "max_bytes": 4096,
        "current_entries": 0,
        "current_bytes": 0,
    }
    tm.remember(["call_a"], "hello")
    s = tm.stats()
    assert s["current_entries"] == 1
    assert s["current_bytes"] == len(b"hello")
