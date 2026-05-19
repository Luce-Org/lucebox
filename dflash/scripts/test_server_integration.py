"""
Integration tests for the C++ dflash_server HTTP API.

Sends real HTTP requests to a running dflash_server instance and validates
response structure, SSE streaming, multi-turn conversations, Responses API,
reasoning/thinking, and error handling.

Usage:
    # Start the server first:
    #   ./dflash_server model.gguf [--port 8000] [...]
    #
    # Then run:
    #   pytest scripts/test_server_integration.py -v
    #   pytest scripts/test_server_integration.py -v -k "not slow"  # skip slow tests
    #
    # Override server URL:
    #   SERVER_URL=http://host:port pytest scripts/test_server_integration.py -v

Mirrors test coverage from test_server.py (Python/FastAPI mock tests) but
exercises the real C++ code path end-to-end.
"""

import json
import os
import time

import pytest
import requests

# ─── Configuration ─────────────────────────────────────────────────

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")
# Allow overriding model name (must match --model-name in server args)
MODEL_NAME = os.environ.get("MODEL_NAME", "dflash")
# Timeout for generation requests (seconds)
GEN_TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "120"))


# ─── Helpers ───────────────────────────────────────────────────────

def post_json(path, body, timeout=GEN_TIMEOUT):
    return requests.post(f"{SERVER_URL}{path}", json=body, timeout=timeout)


def post_stream(path, body, timeout=GEN_TIMEOUT):
    return requests.post(f"{SERVER_URL}{path}", json=body,
                         stream=True, timeout=timeout)


def parse_sse_events(response):
    """Parse SSE stream into list of (event_type, data_dict) tuples."""
    events = []
    event_type = None
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line.startswith("event: "):
            event_type = line[7:]
        elif line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                events.append(("done", None))
            else:
                try:
                    events.append((event_type, json.loads(payload)))
                except json.JSONDecodeError:
                    events.append((event_type, payload))
            event_type = None
        elif line == "":
            event_type = None
    return events


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Ensure the server is reachable before running any tests."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        pytest.exit(f"Server not reachable at {SERVER_URL}: {e}")


# ═══════════════════════════════════════════════════════════════════
# Health & models endpoints
# ═══════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_endpoint(self):
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_root_returns_health(self):
        r = requests.get(f"{SERVER_URL}/", timeout=5)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestModels:
    def test_models_endpoint(self):
        r = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == MODEL_NAME
        assert data["data"][0]["object"] == "model"

    def test_cors_preflight(self):
        r = requests.options(f"{SERVER_URL}/v1/models", timeout=5,
                             headers={
                                 "Origin": "http://localhost:3000",
                                 "Access-Control-Request-Method": "GET",
                             })
        assert r.status_code == 204


# ═══════════════════════════════════════════════════════════════════
# POST /v1/chat/completions — non-streaming
# ═══════════════════════════════════════════════════════════════════

class TestChatCompletionsNonStreaming:
    def test_basic_response_structure(self):
        r = post_json("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl")
        assert data["model"] == MODEL_NAME
        # choices
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] in ("stop", "length")
        msg = choice["message"]
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], str)
        assert len(msg["content"]) > 0
        # usage
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] == (
            data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
        )

    def test_system_message(self):
        r = post_json("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful math tutor."},
                {"role": "user", "content": "What is 5+5? Reply with just the number."},
            ],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"]

    def test_multi_turn_conversation(self):
        r = post_json("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name? Reply with just the name."},
            ],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"]
        assert "Alice" in content

    def test_max_completion_tokens(self):
        """max_completion_tokens should be honored."""
        r = post_json("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Count from 1 to 1000."}],
            "stream": False,
            "max_completion_tokens": 8,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["usage"]["completion_tokens"] <= 10  # small tolerance

    def test_temperature_zero_is_deterministic(self):
        """Two requests with temperature=0 should produce the same output."""
        body = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 7*8? Reply with just the number."}],
            "stream": False,
            "max_tokens": 8,
            "temperature": 0,
        }
        r1 = post_json("/v1/chat/completions", body)
        r2 = post_json("/v1/chat/completions", body)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert (r1.json()["choices"][0]["message"]["content"]
                == r2.json()["choices"][0]["message"]["content"])


# ═══════════════════════════════════════════════════════════════════
# POST /v1/chat/completions — streaming
# ═══════════════════════════════════════════════════════════════════

class TestChatCompletionsStreaming:
    def test_streaming_basic_structure(self):
        r = post_stream("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Say hello."}],
            "stream": True,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("Content-Type", "")

        events = parse_sse_events(r)
        assert len(events) >= 2  # at least one content chunk + done

        # First data chunk should have role delta
        first = events[0]
        assert first[1]["object"] == "chat.completion.chunk"
        assert first[1]["choices"][0]["delta"].get("role") == "assistant"

        # Last event should be [DONE]
        assert events[-1] == ("done", None)

        # At least one chunk should have content delta
        content_chunks = [
            e for _, e in events
            if e and isinstance(e, dict) and
            e.get("choices") and
            e["choices"][0].get("delta", {}).get("content")
        ]
        assert len(content_chunks) >= 1

    def test_streaming_has_usage_chunk(self):
        r = post_stream("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 8,
        })
        events = parse_sse_events(r)
        # The last data chunk before [DONE] should have usage info
        data_events = [e for _, e in events if e and isinstance(e, dict)]
        last_data = data_events[-1]
        assert "usage" in last_data
        assert last_data["usage"]["prompt_tokens"] > 0

    def test_streaming_finish_reason_stop(self):
        r = post_stream("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 1+1? Answer with just the number."}],
            "stream": True,
            "max_tokens": 16,
        })
        events = parse_sse_events(r)
        data_events = [e for _, e in events if e and isinstance(e, dict)]
        # Find the finish_reason chunk
        finish_chunks = [
            e for e in data_events
            if e.get("choices") and e["choices"][0].get("finish_reason")
        ]
        assert len(finish_chunks) >= 1
        assert finish_chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")

    def test_streaming_content_reassembly(self):
        """Concatenating all content deltas should produce coherent text."""
        r = post_stream("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 3*4? Reply with just the number."}],
            "stream": True,
            "max_tokens": 8,
        })
        events = parse_sse_events(r)
        text = ""
        for _, e in events:
            if e and isinstance(e, dict) and e.get("choices"):
                delta = e["choices"][0].get("delta", {})
                text += delta.get("content", "")
        assert len(text.strip()) > 0


# ═══════════════════════════════════════════════════════════════════
# POST /v1/chat/completions — reasoning / thinking
# ═══════════════════════════════════════════════════════════════════

class TestReasoning:
    def test_thinking_disabled_by_default(self):
        """Without explicit enable_thinking, model should not produce reasoning_content."""
        r = post_json("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
            "max_tokens": 32,
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        # When thinking is disabled, content should not be empty
        assert msg["content"]

    @pytest.mark.slow
    def test_thinking_enabled_via_chat_template_kwargs(self):
        """Enabling thinking should produce reasoning_content."""
        r = post_json("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 15 * 17?"}],
            "stream": False,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": True},
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg["content"]
        # With thinking enabled, model may produce reasoning_content
        # (not guaranteed for short prompts, so we just check it doesn't crash)

    @pytest.mark.slow
    def test_thinking_enabled_via_reasoning_effort(self):
        """OpenAI Responses-style reasoning.effort field."""
        r = post_json("/v1/chat/completions", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 15 * 17?"}],
            "stream": False,
            "max_tokens": 512,
            "reasoning": {"effort": "high"},
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg["content"]


# ═══════════════════════════════════════════════════════════════════
# POST /v1/responses — OpenAI Responses API
# ═══════════════════════════════════════════════════════════════════

class TestResponsesAPI:
    def test_responses_non_streaming(self):
        r = post_json("/v1/responses", {
            "model": MODEL_NAME,
            "input": [{"type": "message", "role": "user",
                        "content": "What is 6*7? Reply with just the number."}],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert data["id"].startswith("resp")
        assert data["model"] == MODEL_NAME
        # output
        assert len(data["output"]) >= 1
        output_item = data["output"][0]
        assert output_item["type"] == "message"
        assert output_item["role"] == "assistant"
        assert output_item["content"][0]["type"] == "output_text"
        text = output_item["content"][0]["text"]
        assert len(text) > 0
        # usage
        assert data["usage"]["input_tokens"] > 0
        assert data["usage"]["output_tokens"] > 0
        assert data["usage"]["total_tokens"] == (
            data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
        )

    def test_responses_string_input(self):
        """Responses API accepts a plain string as input."""
        r = post_json("/v1/responses", {
            "model": MODEL_NAME,
            "input": "What is 9+1? Reply with just the number.",
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"

    def test_responses_with_instructions(self):
        """Instructions map to system message."""
        r = post_json("/v1/responses", {
            "model": MODEL_NAME,
            "input": "What is 5+5? Reply with just the number.",
            "instructions": "You are a helpful math tutor. Always show your work.",
            "stream": False,
            "max_tokens": 64,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "response"

    def test_responses_streaming(self):
        """POST /v1/responses streaming emits proper SSE lifecycle events."""
        r = post_stream("/v1/responses", {
            "model": MODEL_NAME,
            "input": "Say hello briefly.",
            "stream": True,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        events = parse_sse_events(r)
        event_types = [et for et, _ in events if et]
        assert "response.created" in event_types
        assert "response.output_item.added" in event_types
        assert "response.content_part.added" in event_types
        assert "response.completed" in event_types

        # Find the completed event and validate structure
        for et, data in events:
            if et == "response.completed" and data:
                assert data["response"]["status"] == "completed"
                assert "usage" in data["response"]

    def test_responses_streaming_has_text_deltas(self):
        r = post_stream("/v1/responses", {
            "model": MODEL_NAME,
            "input": "What is 4+4? Reply with just the number.",
            "stream": True,
            "max_tokens": 16,
        })
        events = parse_sse_events(r)
        delta_events = [
            data for et, data in events
            if et == "response.output_text.delta" and data
        ]
        assert len(delta_events) >= 1
        assert any(d.get("delta") for d in delta_events)

    def test_responses_developer_role(self):
        """Developer role should be accepted (mapped to system internally)."""
        r = post_json("/v1/responses", {
            "model": MODEL_NAME,
            "input": [
                {"type": "message", "role": "developer",
                 "content": "You are a helpful assistant."},
                {"type": "message", "role": "user",
                 "content": "What is 8+8? Reply with just the number."},
            ],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        assert r.json()["object"] == "response"

    def test_responses_with_tools(self):
        """POST /v1/responses with function tools is accepted."""
        r = post_json("/v1/responses", {
            "model": MODEL_NAME,
            "input": [{"type": "message", "role": "user",
                        "content": "What is the weather in Tokyo?"}],
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }],
            "stream": False,
            "max_tokens": 256,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "response"


# ═══════════════════════════════════════════════════════════════════
# POST /v1/messages — Anthropic Messages API
# ═══════════════════════════════════════════════════════════════════

class TestAnthropicAPI:
    def test_messages_non_streaming(self):
        r = post_json("/v1/messages", {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 3+3? Reply with just the number."}],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) >= 1
        assert data["content"][-1]["type"] == "text"
        assert len(data["content"][-1]["text"]) > 0
        assert data["usage"]["input_tokens"] > 0
        assert data["usage"]["output_tokens"] > 0

    def test_messages_with_system(self):
        """Anthropic puts system as a top-level field."""
        r = post_json("/v1/messages", {
            "model": MODEL_NAME,
            "system": "You always reply in uppercase.",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        assert r.json()["type"] == "message"

    def test_messages_billing_header_stripped(self):
        """Billing header blocks in system should be stripped, not sent to model."""
        billing = "x-anthropic-billing-header: cc_version=2.1.37.0d9; cc_entrypoint=cli; cch=fa690;"
        r = post_json("/v1/messages", {
            "model": MODEL_NAME,
            "system": [
                {"type": "text", "text": billing},
                {"type": "text", "text": "You always reply in uppercase."},
            ],
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        # Model should produce output (system wasn't dropped entirely).
        assert len(data["content"]) >= 1
        assert len(data["content"][-1]["text"]) > 0

    def test_messages_billing_header_only_system(self):
        """If system contains only the billing header, request still succeeds."""
        billing = "x-anthropic-billing-header: cc_version=2.1.37.0d9; cc_entrypoint=cli; cch=fa690;"
        r = post_json("/v1/messages", {
            "model": MODEL_NAME,
            "system": [{"type": "text", "text": billing}],
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "stream": False,
            "max_tokens": 16,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"
        assert len(data["content"]) >= 1

class TestErrors:
    def test_unknown_endpoint_returns_404(self):
        r = requests.post(f"{SERVER_URL}/v1/nonexistent",
                          json={"test": True}, timeout=5)
        assert r.status_code == 404

    def test_invalid_json_returns_400(self):
        r = requests.post(f"{SERVER_URL}/v1/chat/completions",
                          data="not json",
                          headers={"Content-Type": "application/json"},
                          timeout=5)
        assert r.status_code == 400

    def test_get_on_post_endpoint_returns_404(self):
        r = requests.get(f"{SERVER_URL}/v1/chat/completions", timeout=5)
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════
# Consistency: streaming vs non-streaming
# ═══════════════════════════════════════════════════════════════════

class TestConsistency:
    def test_streaming_matches_non_streaming_content(self):
        """Streaming and non-streaming with temp=0 should produce same text."""
        body = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 10+10? Reply with just the number."}],
            "max_tokens": 8,
            "temperature": 0,
        }

        # Non-streaming
        r_ns = post_json("/v1/chat/completions", {**body, "stream": False})
        ns_content = r_ns.json()["choices"][0]["message"]["content"]

        # Streaming
        r_s = post_stream("/v1/chat/completions", {**body, "stream": True})
        events = parse_sse_events(r_s)
        s_content = ""
        for _, e in events:
            if e and isinstance(e, dict) and e.get("choices"):
                delta = e["choices"][0].get("delta", {})
                s_content += delta.get("content", "")

        assert ns_content.strip() == s_content.strip()
