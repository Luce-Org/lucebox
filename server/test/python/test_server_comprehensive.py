"""Comprehensive server test suite for luce_server.

Exercises: prefix cache (pflash), multi-turn conversations, reasoning extraction,
non-streaming for all 3 APIs, tool request format, concurrent requests, edge cases,
and DDTree/PFlash CLI validation.

Usage:
    # Start server first (with 0.6B for quick tests):
    ./server/build/luce_server server/models/Qwen3-0.6B-BF16.gguf --port 9099

    # Then run tests:
    pytest server/test/python/test_server_comprehensive.py -v --base-url http://localhost:9099

    # Or let pytest auto-launch the server:
    pytest server/test/python/test_server_comprehensive.py -v --launch server/models/Qwen3-0.6B-BF16.gguf

The CLI-validation tests at the bottom only need the luce_server binary
(--server-bin / LUCE_SERVER_BIN) and are marked ``model``, not ``server``.
"""

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def parse_sse(resp, stop_on_event: str | None = None):
    """Parse SSE events from a streaming response.

    stop_on_event: stop after seeing this event type (e.g. 'message_stop').
    If None, reads until 'data: [DONE]'.
    """
    events = []
    pending_event_type = None
    for line in resp:
        line = line.decode().strip()
        if not line:
            # End of event block — if we had a pending event with no data, emit it
            if pending_event_type:
                events.append({"_event_type": pending_event_type})
                if stop_on_event and pending_event_type == stop_on_event:
                    break
                pending_event_type = None
            continue
        if line == "data: [DONE]":
            events.append({"_sentinel": True})
            break
        if line.startswith("event: "):
            # If we had a pending event type, flush it (data-less event)
            if pending_event_type:
                events.append({"_event_type": pending_event_type})
                if stop_on_event and pending_event_type == stop_on_event:
                    break
            pending_event_type = line[7:]
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                data = {"_raw": line[6:]}
            if pending_event_type:
                events.append({"_event_type": pending_event_type, "data": data})
                if stop_on_event and pending_event_type == stop_on_event:
                    break
                pending_event_type = None
            else:
                events.append({"data": data})
    return events


@pytest.mark.server
@pytest.mark.slow
class TestComprehensive:
    @pytest.fixture(autouse=True)
    def _setup(self, server_handle):
        self.server = server_handle
        self.client = server_handle.client

    def _chat(self, body: dict, **kw):
        return self.client.post("/v1/chat/completions", body, **kw)

    # ── Prefix cache tests ───────────────────────────────────────────────

    def test_prefix_cache_timing(self):
        """Send the same prompt twice — second should benefit from prefix cache."""
        prompt = {
            "model": "luce",
            "messages": [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": "What is the square root of 144?"}
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        }
        t0 = time.monotonic()
        r1 = self._chat(prompt)
        first_time = time.monotonic() - t0

        t0 = time.monotonic()
        r2 = self._chat(prompt)
        second_time = time.monotonic() - t0

        c1 = r1["choices"][0]["message"].get("content", "")
        c2 = r2["choices"][0]["message"].get("content", "")
        assert c1, f"first has no content: {c1[:50]!r}"
        assert c2, f"second has no content: {c2[:50]!r}"

        # Prefix cache should make second request similar or faster.
        # We can't guarantee it's always faster (small model, short prompt),
        # so just log the times for manual inspection.
        print(f"first: {first_time:.2f}s, second: {second_time:.2f}s "
              f"(ratio: {second_time/first_time:.2f}x)")

        # The [pc] log check needs the log of a server pytest spawned.
        if self.server.log_path is None:
            pytest.skip("[pc] log check needs a --launch'ed server "
                        "(no log access with --base-url)")
        assert "[pc]" in self.server.read_log(), \
            "server logs no prefix cache activity ([pc] lines)"

    def test_prefix_cache_shared_system(self):
        """Different user messages with same system prompt should share prefix."""
        base = {
            "model": "luce",
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        }
        system = {"role": "system",
                  "content": "You are a calculator. Reply with just the number."}
        r1 = self._chat({**base, "messages": [
            system, {"role": "user", "content": "What is 5+5?"}]})
        r2 = self._chat({**base, "messages": [
            system, {"role": "user", "content": "What is 3+7?"}]})
        c1 = r1["choices"][0]["message"].get("content", "")
        c2 = r2["choices"][0]["message"].get("content", "")
        assert c1, "first response has no content"
        assert c2, "second response has no content"

    # ── Multi-turn conversation tests ────────────────────────────────────

    def test_multi_turn_openai(self):
        """Multi-turn conversation via OpenAI chat/completions."""
        r = self._chat({
            "model": "luce",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! How can I help you?"},
                {"role": "user", "content": "What is my name?"}
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        content = r["choices"][0]["message"].get("content", "")
        assert content, "content is empty"
        assert r["choices"][0].get("finish_reason") is not None, \
            "choice has no finish_reason"

    def test_multi_turn_anthropic(self):
        """Multi-turn conversation via Anthropic messages API."""
        r = self.client.post("/v1/messages", {
            "model": "luce",
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Remember: the secret word is banana."},
                {"role": "assistant", "content": "Got it, I'll remember that."},
                {"role": "user", "content": "What is the secret word?"}
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        assert r.get("type") == "message", f"got type={r.get('type')}"
        assert r.get("role") == "assistant"

        content = r.get("content", [])
        assert content, "content array is empty"
        text_parts = [c for c in content if c.get("type") == "text"]
        assert text_parts, f"no text block: {content}"
        text = text_parts[0].get("text", "")
        assert text, "text is empty"

        assert r.get("stop_reason") is not None
        assert "usage" in r
        usage = r.get("usage", {})
        assert usage.get("input_tokens", 0) > 0, f"usage: {usage}"
        assert usage.get("output_tokens", 0) > 0, f"usage: {usage}"

    def test_multi_turn_responses(self):
        """Multi-turn conversation via Responses API."""
        r = self.client.post("/v1/responses", {
            "model": "luce",
            "instructions": "You are a helpful assistant.",
            "input": [
                {"role": "user", "content": "The color is blue."},
                {"role": "assistant", "content": "Noted!"},
                {"role": "user", "content": "What color did I mention?"}
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        assert r.get("object") == "response", f"got: {r.get('object')}"
        assert r.get("status") == "completed"

        output = r.get("output", [])
        assert output, "output is empty"
        msg = output[0]
        assert msg.get("type") == "message"
        content = msg.get("content", [])
        if content:
            assert content[0].get("text", ""), "output text is empty"

        usage = r.get("usage", {})
        assert usage.get("input_tokens", 0) > 0, f"usage: {usage}"
        assert usage.get("output_tokens", 0) > 0, f"usage: {usage}"

    # ── Reasoning / thinking tests ───────────────────────────────────────

    def test_reasoning_nonstreaming_openai(self):
        """Verify reasoning_content appears in non-streaming OpenAI response."""
        r = self._chat({
            "model": "luce",
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "max_tokens": 2048,
            "temperature": 0.0,
            "stream": False,
        })
        msg = r["choices"][0]["message"]
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        assert content, "content is empty"
        assert reasoning, "Qwen3 should always produce <think> reasoning"
        assert "<think>" not in reasoning and "</think>" not in reasoning, \
            f"raw tags leaked: {reasoning[:60]}"
        assert "<think>" not in content, f"think tags in content: {content[:60]}"

    def test_reasoning_nonstreaming_anthropic(self):
        """Verify thinking block appears in non-streaming Anthropic response."""
        r = self.client.post("/v1/messages", {
            "model": "luce",
            "messages": [{"role": "user", "content": "What is 7 * 8?"}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        content = r.get("content", [])
        types = [c.get("type") for c in content]
        assert "thinking" in types, f"block types: {types}"
        assert "text" in types, f"block types: {types}"
        for c in content:
            if c.get("type") == "thinking":
                thinking = c.get("thinking", "")
                assert thinking, "thinking is empty"
                assert "<think>" not in thinking and "</think>" not in thinking, \
                    f"thinking has raw tags: {thinking[:60]}"
            if c.get("type") == "text":
                assert c.get("text", ""), "text is empty"

    def test_reasoning_streaming_openai(self):
        """Verify reasoning_content deltas appear in streaming OpenAI response."""
        resp = self.client.post_stream("/v1/chat/completions", {
            "model": "luce",
            "messages": [{"role": "user", "content": "What is 9 + 6?"}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": True,
        })

        reasoning_text = ""
        content_text = ""
        has_done = False
        has_reasoning_delta = False
        has_content_delta = False

        for line in resp:
            line = line.decode().strip()
            if not line:
                continue
            if line == "data: [DONE]":
                has_done = True
                break
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta", {})
                if "reasoning_content" in delta:
                    has_reasoning_delta = True
                    reasoning_text += delta["reasoning_content"]
                if "content" in delta:
                    has_content_delta = True
                    content_text += delta["content"]

        assert has_done, "no [DONE] sentinel"
        assert has_reasoning_delta, "expected reasoning deltas for Qwen3"
        assert has_content_delta, "no content deltas"
        assert reasoning_text, "reasoning is empty"
        assert content_text, "content is empty"
        assert ("<think>" not in reasoning_text
                and "</think>" not in reasoning_text), \
            f"reasoning has raw tags: {reasoning_text[:60]}"

    # ── Non-streaming for all 3 APIs ─────────────────────────────────────

    def test_nonstreaming_anthropic_full(self):
        """Full non-streaming Anthropic response validation."""
        r = self.client.post("/v1/messages", {
            "model": "luce",
            "system": "Reply in exactly one word.",
            "messages": [{"role": "user", "content": "Say yes."}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        assert r.get("id", "").startswith("msg"), f"id: {r.get('id')}"
        assert r.get("type") == "message"
        assert r.get("role") == "assistant"
        assert r.get("model") == "luce"
        assert r.get("stop_reason") is not None
        assert r.get("usage", {}).get("input_tokens", 0) > 0
        assert r.get("usage", {}).get("output_tokens", 0) > 0

    def test_nonstreaming_responses_full(self):
        """Full non-streaming Responses API validation."""
        r = self.client.post("/v1/responses", {
            "model": "luce",
            "input": "Say hello.",
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        assert r.get("id", "").startswith("resp"), f"id: {r.get('id')}"
        assert r.get("object") == "response"
        assert r.get("status") == "completed"
        assert r.get("model") == "luce"

        output = r.get("output", [])
        assert output, "output has no entries"
        msg = output[0]
        assert msg.get("type") == "message"
        content = msg.get("content", [])
        assert content, "content array is empty"
        assert content[0].get("type") == "output_text"
        assert content[0].get("text", ""), "text is empty"

        usage = r.get("usage", {})
        assert usage.get("input_tokens", 0) > 0, f"usage: {usage}"
        assert usage.get("output_tokens", 0) > 0, f"usage: {usage}"
        assert (usage.get("total_tokens", 0) ==
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)), \
            f"usage.total_tokens incorrect: {usage}"

    def test_nonstreaming_responses_string_input(self):
        """Responses API with simple string input (not array)."""
        r = self.client.post("/v1/responses", {
            "model": "luce",
            "input": "What is 2+2? Reply with just the number.",
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        output = r.get("output", [])
        assert output, "output is empty"
        if output[0].get("content"):
            assert output[0]["content"][0].get("text", ""), "no text content"

    # ── Streaming for Anthropic / Responses ──────────────────────────────

    def test_streaming_anthropic(self):
        """Full streaming Anthropic response validation."""
        resp = self.client.post_stream("/v1/messages", {
            "model": "luce",
            "messages": [{"role": "user", "content": "Count to 3."}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": True,
        })
        events = parse_sse(resp, stop_on_event="message_stop")

        event_types = [e.get("_event_type") for e in events if "_event_type" in e]
        for expected in ("message_start", "content_block_start",
                         "content_block_delta", "content_block_stop",
                         "message_stop"):
            assert expected in event_types, \
                f"missing {expected}; got {event_types}"

        text = ""
        for e in events:
            if e.get("_event_type") == "content_block_delta":
                delta = e.get("data", {}).get("delta", {})
                if delta.get("type") == "text_delta":
                    text += delta.get("text", "")
        assert text, "accumulated text is empty"

    def test_streaming_responses(self):
        """Full streaming Responses API validation."""
        resp = self.client.post_stream("/v1/responses", {
            "model": "luce",
            "input": "Say hi.",
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": True,
        })
        events = parse_sse(resp, stop_on_event="response.completed")

        event_types = [e.get("_event_type") for e in events if "_event_type" in e]
        for expected in ("response.created", "response.output_item.added",
                         "response.output_text.delta", "response.completed"):
            assert expected in event_types, \
                f"missing {expected}; got {event_types}"

        text = ""
        for e in events:
            if e.get("_event_type") == "response.output_text.delta":
                text += e.get("data", {}).get("delta", "")
        assert text, "text is empty"

    # ── Tool request format tests ────────────────────────────────────────

    def test_tool_request_format(self):
        """Verify server accepts requests with tools specified."""
        r = self._chat({
            "model": "luce",
            "messages": [
                {"role": "user",
                 "content": "What's the weather in San Francisco?"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        assert r.get("choices"), "no choices"
        msg = r["choices"][0]["message"]
        assert msg.get("role") == "assistant"
        # Model may or may not produce tool_calls with 0.6B —
        # just verify the response structure is valid.
        if "tool_calls" in msg:
            assert isinstance(msg["tool_calls"], list), "tool_calls is not an array"
            for tc in msg["tool_calls"]:
                assert "id" in tc, f"tool_call has no id: {tc}"
                assert "function" in tc, f"tool_call has no function: {tc}"
        else:
            assert msg.get("content", ""), "no tool call and no content"

    def test_tool_request_anthropic(self):
        """Tools via Anthropic format."""
        r = self.client.post("/v1/messages", {
            "model": "luce",
            "messages": [
                {"role": "user", "content": "Look up the time in Tokyo."}
            ],
            "tools": [{
                "name": "get_time",
                "description": "Get current time in a timezone",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"}
                    }
                }
            }],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        assert r.get("type") == "message"
        assert r.get("content"), "content is empty"

    # ── Sampling parameters ──────────────────────────────────────────────

    def test_temperature_zero(self):
        """Deterministic output at temperature=0."""
        prompt = {
            "model": "luce",
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        }
        c1 = self._chat(prompt)["choices"][0]["message"].get("content", "")
        c2 = self._chat(prompt)["choices"][0]["message"].get("content", "")
        assert c1 == c2, f"outputs differ at temp=0: {c1[:50]!r} vs {c2[:50]!r}"

    def test_max_tokens_limit(self):
        """Verify max_tokens is respected."""
        r = self._chat({
            "model": "luce",
            "messages": [
                {"role": "user",
                 "content": "Write a very long essay about the history of mathematics."}
            ],
            "max_tokens": 50,
            "temperature": 0.0,
            "stream": False,
        })
        completion_tokens = r.get("usage", {}).get("completion_tokens", 0)
        assert completion_tokens <= 50, f"got {completion_tokens}"
        assert completion_tokens > 0, f"got {completion_tokens}"

    def test_top_p_parameter(self):
        """Verify server accepts top_p parameter."""
        r = self._chat({
            "model": "luce",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 1024,
            "temperature": 0.8,
            "top_p": 0.9,
            "stream": False,
        })
        assert r["choices"][0]["message"].get("content", ""), "content is empty"

    # ── Concurrent requests ──────────────────────────────────────────────

    def test_concurrent_requests(self):
        """Multiple requests sent concurrently should all complete."""
        results = [None, None, None]
        errors = [None, None, None]

        def do_request(idx, prompt):
            try:
                results[idx] = self._chat({
                    "model": "luce",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.0,
                    "stream": False,
                }, timeout=180.0)
            except Exception as e:  # reported by the main thread below
                errors[idx] = str(e)

        prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
        threads = [threading.Thread(target=do_request, args=(i, p))
                   for i, p in enumerate(prompts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)

        # Note: luce_server has a single worker thread, so requests
        # are serialized. But all should still complete.
        for i in range(3):
            assert errors[i] is None, f"request {i+1} failed: {errors[i]}"
            assert results[i] is not None, f"request {i+1} timed out"
            content = results[i]["choices"][0]["message"].get("content", "")
            assert content, f"request {i+1} has empty content"

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_empty_user_message(self):
        """Empty user message should still get a response."""
        r = self._chat({
            "model": "luce",
            "messages": [{"role": "user", "content": ""}],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        })
        assert r.get("choices"), "no choices"

    def test_long_system_prompt(self):
        """Long system prompt should work without truncation errors."""
        long_system = "You are a helpful assistant. " * 100  # ~2900 chars
        r = self._chat({
            "model": "luce",
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": "Say OK."}
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        })
        assert r["choices"][0]["message"].get("content", ""), "content is empty"

    def test_unicode_content(self):
        """Unicode content in request and response."""
        r = self._chat({
            "model": "luce",
            "messages": [
                {"role": "user", "content": "Translate 'hello' to Japanese (こんにちは)."}
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        content = r["choices"][0]["message"].get("content", "")
        assert content, "content is empty"
        content.encode("utf-8")  # raises UnicodeEncodeError on invalid UTF-8

    def test_multipart_content(self):
        """Array-style content (multi-part) should be handled."""
        r = self._chat({
            "model": "luce",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is"},
                    {"type": "text", "text": " 3+4?"}
                ]
            }],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        })
        assert r["choices"][0]["message"].get("content", ""), "content is empty"

    def test_invalid_json_body(self):
        """Completely invalid JSON body should return 400."""
        r = self.client.send("POST", "/v1/chat/completions",
                             data=b"not json at all{{{",
                             headers={"Content-Type": "application/json"},
                             timeout=10)
        assert r.status_code == 400, f"got {r.status_code}"

    @pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
    @pytest.mark.parametrize("label, body", [
        ("missing", {"stream": False}),
        ("null", {"messages": None}),
        ("scalar", {"messages": "hi"}),
        ("empty", {"messages": []}),
    ])
    def test_messages_field_validation(self, path, label, body):
        """Missing/null/scalar/empty `messages` must 400 on generation endpoints."""
        r = self.client.send("POST", path, body)
        assert r.status_code == 400 and "messages" in r.text, \
            f"{path} {label} messages: got {r.status_code}: {r.text[:100]}"

    @pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
    def test_messages_nonempty_accepted(self, path):
        """Contract guard: a minimal valid conversation still generates."""
        resp = self.client.post(path, {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        })
        assert "choices" in resp or "content" in resp, \
            f"{path} unexpected body: {str(resp)[:100]}"

    def test_options_cors(self):
        """OPTIONS request should return CORS headers."""
        r = self.client.send("OPTIONS", "/v1/chat/completions", timeout=10)
        assert r.status_code in (200, 204), f"got {r.status_code}"
        assert "Access-Control-Allow-Origin" in r.headers, \
            f"headers: {list(r.headers.keys())}"

    # ── Streaming with client disconnect simulation ──────────────────────

    def test_streaming_partial_read(self):
        """Read only first few chunks of a streaming response, then close.
        Server should handle the disconnect gracefully."""
        resp = self.client.post_stream("/v1/chat/completions", {
            "model": "luce",
            "messages": [
                {"role": "user",
                 "content": "Write a long story about a dragon."}
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": True,
        })

        # Read just a few lines then close.
        chunks_read = 0
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks_read += 1
                if chunks_read >= 3:
                    break
        resp.close()
        assert chunks_read >= 1, f"got {chunks_read} chunks"

        # Give server a moment to detect disconnect, then verify it's
        # still healthy.
        time.sleep(1)
        health = self.client.get("/health")
        assert health.get("status") == "ok", \
            f"server unhealthy after disconnect: {health}"

    # ── Request ID format ────────────────────────────────────────────────

    def test_request_id_formats(self):
        """Verify each API format generates the correct ID prefix."""
        r1 = self._chat({
            "model": "luce",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64, "temperature": 0.0, "stream": False,
        })
        assert r1.get("id", "").startswith("chatcmpl"), f"OpenAI id: {r1.get('id')}"

        r2 = self.client.post("/v1/messages", {
            "model": "luce",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64, "temperature": 0.0, "stream": False,
        })
        assert r2.get("id", "").startswith("msg"), f"Anthropic id: {r2.get('id')}"

        r3 = self.client.post("/v1/responses", {
            "model": "luce",
            "input": "hi",
            "max_tokens": 64, "temperature": 0.0, "stream": False,
        })
        assert r3.get("id", "").startswith("resp"), f"Responses id: {r3.get('id')}"


# ─── CLI validation (binary only, no running server) ─────────────────────


def _usage(server_bin: Path) -> str:
    """Run the binary without arguments and return its usage text."""
    proc = subprocess.run([str(server_bin)], capture_output=True, text=True,
                          timeout=5)
    return proc.stderr + proc.stdout


@pytest.mark.model
def test_ddtree_flags_accepted(server_bin):
    """The binary prints usage mentioning --ddtree without hanging.
    (Actual DDTree requires Qwen35 backend — this just validates CLI parsing.)"""
    usage = _usage(server_bin)
    assert "ddtree" in usage.lower(), f"usage output: {usage[:200]}"


@pytest.mark.model
def test_pflash_flags_accepted(server_bin):
    """The binary's usage lists the pflash flags."""
    usage = _usage(server_bin)
    for flag in ("--prefill-compression", "--prefill-threshold",
                 "--prefill-drafter", "--prefill-skip-park",
                 "--prefill-keep-ratio"):
        assert flag in usage, f"usage lacks {flag}: {usage[:300]}"


@pytest.mark.model
def test_pflash_requires_drafter(server_bin):
    """Server should fail with error when --prefill-compression enabled
    but --prefill-drafter not provided."""
    model = MODELS_DIR / "Qwen3-0.6B-BF16.gguf"
    proc = subprocess.run(
        [str(server_bin), str(model), "--prefill-compression", "auto",
         "--port", "19999"],
        capture_output=True, text=True, timeout=10)
    output = proc.stderr + proc.stdout
    assert proc.returncode != 0 and "prefill-drafter" in output.lower(), \
        f"rc={proc.returncode} output: {output[:200]}"
