#!/usr/bin/env python3
"""Comprehensive server test suite for dflash_server.

Exercises: prefix cache (pflash), multi-turn conversations, reasoning extraction,
non-streaming for all 3 APIs, tool request format, concurrent requests, edge cases,
and DDTree startup validation.

Usage:
    # Start server first (with 0.6B for quick tests):
    ./dflash/build/dflash_server dflash/models/Qwen3-0.6B-BF16.gguf --port 9099

    # Then run tests:
    python3 dflash/tests/test_server_comprehensive.py --base-url http://localhost:9099

    # Or auto-launch:
    python3 dflash/tests/test_server_comprehensive.py --launch dflash/models/Qwen3-0.6B-BF16.gguf
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error


# ─── Test framework ──────────────────────────────────────────────────────

class TestSuite:
    def __init__(self, base_url: str, server_log_path: str | None = None):
        self.base = base_url.rstrip("/")
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.server_log = server_log_path

    def _req(self, method: str, path: str, body: dict | None = None,
             stream: bool = False, timeout: float = 120.0):
        url = self.base + path
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if body else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        resp = urllib.request.urlopen(req, timeout=timeout)
        if stream:
            return resp
        return json.loads(resp.read().decode())

    def _check(self, name: str, ok: bool, detail: str = ""):
        if ok:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {detail}")

    def _skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⏭️  {name}: {reason}")

    def _read_server_log(self) -> str:
        """Read server stderr log if available."""
        if self.server_log and os.path.exists(self.server_log):
            with open(self.server_log) as f:
                return f.read()
        return ""

    def _parse_sse(self, resp, stop_on_event: str | None = None):
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
                    events.append({"data": data})
        return events

    # ── Prefix cache tests ───────────────────────────────────────────────

    def test_prefix_cache_timing(self):
        """Send the same prompt twice — second should benefit from prefix cache."""
        print("\n[PC-1] Prefix cache — repeated prompt timing")
        prompt = {
            "model": "dflash",
            "messages": [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": "What is the square root of 144?"}
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            t0 = time.monotonic()
            r1 = self._req("POST", "/v1/chat/completions", prompt)
            t1 = time.monotonic()
            first_time = t1 - t0

            t0 = time.monotonic()
            r2 = self._req("POST", "/v1/chat/completions", prompt)
            t1 = time.monotonic()
            second_time = t1 - t0

            self._check("first request completes", True)
            self._check("second request completes", True)

            c1 = r1["choices"][0]["message"].get("content", "")
            c2 = r2["choices"][0]["message"].get("content", "")
            self._check("first has content", len(c1) > 0, f"content='{c1[:50]}'")
            self._check("second has content", len(c2) > 0, f"content='{c2[:50]}'")

            # Prefix cache should make second request similar or faster.
            # We can't guarantee it's always faster (small model, short prompt),
            # so just log the times for manual inspection.
            print(f"    → first: {first_time:.2f}s, second: {second_time:.2f}s "
                  f"(ratio: {second_time/first_time:.2f}x)")

            # Check server log for prefix cache activity.
            log = self._read_server_log()
            has_pc_log = "[pc]" in log
            self._check("server logs prefix cache activity", has_pc_log,
                         "no [pc] log lines found")

        except Exception as e:
            self._check("prefix cache timing test", False, str(e))

    def test_prefix_cache_shared_system(self):
        """Different user messages with same system prompt should share prefix."""
        print("\n[PC-2] Prefix cache — shared system prompt")
        base = {
            "model": "dflash",
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            r1 = self._req("POST", "/v1/chat/completions", {
                **base,
                "messages": [
                    {"role": "system", "content": "You are a calculator. Reply with just the number."},
                    {"role": "user", "content": "What is 5+5?"}
                ]
            })
            r2 = self._req("POST", "/v1/chat/completions", {
                **base,
                "messages": [
                    {"role": "system", "content": "You are a calculator. Reply with just the number."},
                    {"role": "user", "content": "What is 3+7?"}
                ]
            })
            c1 = r1["choices"][0]["message"].get("content", "")
            c2 = r2["choices"][0]["message"].get("content", "")
            self._check("first response has content", len(c1) > 0)
            self._check("second response has content", len(c2) > 0)
            print(f"    → r1: {c1!r}, r2: {c2!r}")
        except Exception as e:
            self._check("shared system prefix test", False, str(e))

    # ── Multi-turn conversation tests ────────────────────────────────────

    def test_multi_turn_openai(self):
        """Multi-turn conversation via OpenAI chat/completions."""
        print("\n[MT-1] Multi-turn conversation — OpenAI format")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
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
            self._check("got response", True)
            content = r["choices"][0]["message"].get("content", "")
            self._check("content is non-empty", len(content) > 0, f"content='{content[:80]}'")
            self._check("choices has correct structure",
                         r["choices"][0].get("finish_reason") is not None)
            print(f"    → content: {content!r}")
        except Exception as e:
            self._check("multi-turn OpenAI", False, str(e))

    def test_multi_turn_anthropic(self):
        """Multi-turn conversation via Anthropic messages API."""
        print("\n[MT-2] Multi-turn conversation — Anthropic format")
        try:
            r = self._req("POST", "/v1/messages", {
                "model": "dflash",
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
            self._check("got response", True)
            self._check("type is message", r.get("type") == "message",
                         f"got type={r.get('type')}")
            self._check("role is assistant", r.get("role") == "assistant")

            content = r.get("content", [])
            self._check("content array non-empty", len(content) > 0)
            text_parts = [c for c in content if c.get("type") == "text"]
            self._check("has text block", len(text_parts) > 0)
            if text_parts:
                text = text_parts[0].get("text", "")
                self._check("text non-empty", len(text) > 0, f"text='{text[:80]}'")
                print(f"    → text: {text!r}")

            self._check("has stop_reason", r.get("stop_reason") is not None)
            self._check("has usage", "usage" in r)
            usage = r.get("usage", {})
            self._check("usage.input_tokens > 0",
                         usage.get("input_tokens", 0) > 0)
            self._check("usage.output_tokens > 0",
                         usage.get("output_tokens", 0) > 0)
        except Exception as e:
            self._check("multi-turn Anthropic", False, str(e))

    def test_multi_turn_responses(self):
        """Multi-turn conversation via Responses API."""
        print("\n[MT-3] Multi-turn conversation — Responses format")
        try:
            r = self._req("POST", "/v1/responses", {
                "model": "dflash",
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
            self._check("got response", True)
            self._check("object is response", r.get("object") == "response",
                         f"got: {r.get('object')}")
            self._check("status is completed", r.get("status") == "completed")

            output = r.get("output", [])
            self._check("output non-empty", len(output) > 0)
            if output:
                msg = output[0]
                self._check("output type is message", msg.get("type") == "message")
                content = msg.get("content", [])
                if content:
                    text = content[0].get("text", "")
                    self._check("output text non-empty", len(text) > 0)
                    print(f"    → text: {text!r}")

            usage = r.get("usage", {})
            self._check("has input_tokens", usage.get("input_tokens", 0) > 0)
            self._check("has output_tokens", usage.get("output_tokens", 0) > 0)
        except Exception as e:
            self._check("multi-turn Responses", False, str(e))

    # ── Reasoning / thinking tests ───────────────────────────────────────

    def test_reasoning_nonstreaming_openai(self):
        """Verify reasoning_content appears in non-streaming OpenAI response."""
        print("\n[TH-1] Reasoning — non-streaming OpenAI")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [{"role": "user", "content": "What is 2 + 3?"}],
                "max_tokens": 2048,
                "temperature": 0.0,
                "stream": False,
            })
            msg = r["choices"][0]["message"]
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            self._check("has content", len(content) > 0, f"content='{content[:60]}'")
            self._check("has reasoning_content", len(reasoning) > 0,
                         "Qwen3 should always produce <think> reasoning")
            self._check("reasoning doesn't contain <think> tags",
                         "<think>" not in reasoning and "</think>" not in reasoning,
                         f"raw tags leaked: {reasoning[:60]}")
            self._check("content doesn't contain <think> tags",
                         "<think>" not in content,
                         f"think tags in content: {content[:60]}")
            print(f"    → reasoning: {reasoning[:80]!r}...")
            print(f"    → content: {content!r}")
        except Exception as e:
            self._check("reasoning non-streaming OpenAI", False, str(e))

    def test_reasoning_nonstreaming_anthropic(self):
        """Verify thinking block appears in non-streaming Anthropic response."""
        print("\n[TH-2] Reasoning — non-streaming Anthropic")
        try:
            r = self._req("POST", "/v1/messages", {
                "model": "dflash",
                "messages": [{"role": "user", "content": "What is 7 * 8?"}],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": False,
            })
            content = r.get("content", [])
            types = [c.get("type") for c in content]
            self._check("has thinking block", "thinking" in types,
                         f"block types: {types}")
            self._check("has text block", "text" in types)
            for c in content:
                if c.get("type") == "thinking":
                    thinking = c.get("thinking", "")
                    self._check("thinking non-empty", len(thinking) > 0)
                    self._check("thinking has no raw tags",
                                "<think>" not in thinking and "</think>" not in thinking)
                    print(f"    → thinking: {thinking[:80]!r}...")
                if c.get("type") == "text":
                    text = c.get("text", "")
                    self._check("text non-empty", len(text) > 0, f"text='{text[:60]}'")
                    print(f"    → text: {text!r}")
        except Exception as e:
            self._check("reasoning non-streaming Anthropic", False, str(e))

    def test_reasoning_streaming_openai(self):
        """Verify reasoning_content deltas appear in streaming OpenAI response."""
        print("\n[TH-3] Reasoning — streaming OpenAI")
        try:
            resp = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [{"role": "user", "content": "What is 9 + 6?"}],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": True,
            }, stream=True)

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

            self._check("got [DONE]", has_done)
            self._check("has reasoning_content deltas", has_reasoning_delta,
                         "expected reasoning deltas for Qwen3")
            self._check("has content deltas", has_content_delta)
            self._check("reasoning non-empty", len(reasoning_text) > 0)
            self._check("content non-empty", len(content_text) > 0,
                         f"content='{content_text}'")
            self._check("reasoning has no raw tags",
                         "<think>" not in reasoning_text and
                         "</think>" not in reasoning_text)
            print(f"    → reasoning: {reasoning_text[:60]!r}...")
            print(f"    → content: {content_text!r}")
        except Exception as e:
            self._check("reasoning streaming OpenAI", False, str(e))

    # ── Non-streaming for all 3 APIs ─────────────────────────────────────

    def test_nonstreaming_anthropic_full(self):
        """Full non-streaming Anthropic response validation."""
        print("\n[NS-1] Non-streaming — Anthropic full validation")
        try:
            r = self._req("POST", "/v1/messages", {
                "model": "dflash",
                "system": "Reply in exactly one word.",
                "messages": [{"role": "user", "content": "Say yes."}],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": False,
            })
            self._check("id starts with msg", r.get("id", "").startswith("msg"))
            self._check("type is message", r.get("type") == "message")
            self._check("role is assistant", r.get("role") == "assistant")
            self._check("model is dflash", r.get("model") == "dflash")
            self._check("stop_reason present", r.get("stop_reason") is not None)
            self._check("usage.input_tokens present",
                         r.get("usage", {}).get("input_tokens", 0) > 0)
            self._check("usage.output_tokens present",
                         r.get("usage", {}).get("output_tokens", 0) > 0)
        except Exception as e:
            self._check("non-streaming Anthropic", False, str(e))

    def test_nonstreaming_responses_full(self):
        """Full non-streaming Responses API validation."""
        print("\n[NS-2] Non-streaming — Responses full validation")
        try:
            r = self._req("POST", "/v1/responses", {
                "model": "dflash",
                "input": "Say hello.",
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": False,
            })
            self._check("id starts with resp", r.get("id", "").startswith("resp"))
            self._check("object is response", r.get("object") == "response")
            self._check("status is completed", r.get("status") == "completed")
            self._check("model is dflash", r.get("model") == "dflash")

            output = r.get("output", [])
            self._check("output has entries", len(output) > 0)
            if output:
                msg = output[0]
                self._check("first output type is message",
                             msg.get("type") == "message")
                content = msg.get("content", [])
                self._check("content array non-empty", len(content) > 0)
                if content:
                    self._check("content type is output_text",
                                 content[0].get("type") == "output_text")
                    text = content[0].get("text", "")
                    self._check("text non-empty", len(text) > 0)
                    print(f"    → text: {text!r}")

            usage = r.get("usage", {})
            self._check("usage.input_tokens > 0", usage.get("input_tokens", 0) > 0)
            self._check("usage.output_tokens > 0", usage.get("output_tokens", 0) > 0)
            self._check("usage.total_tokens correct",
                         usage.get("total_tokens", 0) ==
                         usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
        except Exception as e:
            self._check("non-streaming Responses", False, str(e))

    def test_nonstreaming_responses_string_input(self):
        """Responses API with simple string input (not array)."""
        print("\n[NS-3] Non-streaming — Responses string input")
        try:
            r = self._req("POST", "/v1/responses", {
                "model": "dflash",
                "input": "What is 2+2? Reply with just the number.",
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": False,
            })
            self._check("got response", True)
            output = r.get("output", [])
            self._check("output non-empty", len(output) > 0)
            if output and output[0].get("content"):
                text = output[0]["content"][0].get("text", "")
                self._check("got text content", len(text) > 0)
                print(f"    → text: {text!r}")
        except Exception as e:
            self._check("Responses string input", False, str(e))

    # ── Streaming for Anthropic / Responses ──────────────────────────────

    def test_streaming_anthropic(self):
        """Full streaming Anthropic response validation."""
        print("\n[ST-1] Streaming — Anthropic full validation")
        try:
            resp = self._req("POST", "/v1/messages", {
                "model": "dflash",
                "messages": [{"role": "user", "content": "Count to 3."}],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": True,
            }, stream=True)

            events = self._parse_sse(resp, stop_on_event="message_stop")

            event_types = [e.get("_event_type") for e in events if "_event_type" in e]
            self._check("has message_start", "message_start" in event_types)
            self._check("has content_block_start", "content_block_start" in event_types)
            self._check("has content_block_delta", "content_block_delta" in event_types)
            self._check("has content_block_stop", "content_block_stop" in event_types)
            self._check("has message_stop", "message_stop" in event_types)

            # Extract text from deltas
            text = ""
            for e in events:
                if e.get("_event_type") == "content_block_delta":
                    delta = e.get("data", {}).get("delta", {})
                    if delta.get("type") == "text_delta":
                        text += delta.get("text", "")
            self._check("accumulated text non-empty", len(text) > 0,
                         f"text='{text}'")
            print(f"    → text: {text!r}")

        except Exception as e:
            self._check("streaming Anthropic", False, str(e))

    def test_streaming_responses(self):
        """Full streaming Responses API validation."""
        print("\n[ST-2] Streaming — Responses full validation")
        try:
            resp = self._req("POST", "/v1/responses", {
                "model": "dflash",
                "input": "Say hi.",
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": True,
            }, stream=True)

            events = self._parse_sse(resp, stop_on_event="response.completed")

            event_types = [e.get("_event_type") for e in events if "_event_type" in e]
            self._check("has response.created", "response.created" in event_types)
            self._check("has response.output_item.added",
                         "response.output_item.added" in event_types)
            self._check("has response.output_text.delta",
                         "response.output_text.delta" in event_types)
            self._check("has response.completed", "response.completed" in event_types)

            text = ""
            for e in events:
                if e.get("_event_type") == "response.output_text.delta":
                    text += e.get("data", {}).get("delta", "")
            self._check("text non-empty", len(text) > 0, f"text='{text}'")
            print(f"    → text: {text!r}")

        except Exception as e:
            self._check("streaming Responses", False, str(e))

    # ── Tool request format tests ────────────────────────────────────────

    def test_tool_request_format(self):
        """Verify server accepts requests with tools specified."""
        print("\n[TL-1] Tool request format — tools in request body")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
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
            self._check("got response", True)
            self._check("has choices", len(r.get("choices", [])) > 0)
            msg = r["choices"][0]["message"]
            self._check("message has role", msg.get("role") == "assistant")
            # Model may or may not produce tool_calls with 0.6B —
            # just verify the response structure is valid.
            if "tool_calls" in msg:
                self._check("tool_calls is array", isinstance(msg["tool_calls"], list))
                for tc in msg["tool_calls"]:
                    self._check("tool_call has id", "id" in tc)
                    self._check("tool_call has function", "function" in tc)
                print(f"    → tool_calls: {json.dumps(msg['tool_calls'])[:100]}")
            else:
                content = msg.get("content", "")
                self._check("has content (no tool call)", len(content) > 0)
                print(f"    → content (no tool call): {content[:80]!r}")
        except Exception as e:
            self._check("tool request format", False, str(e))

    def test_tool_request_anthropic(self):
        """Tools via Anthropic format."""
        print("\n[TL-2] Tool request format — Anthropic")
        try:
            r = self._req("POST", "/v1/messages", {
                "model": "dflash",
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
            self._check("got valid response", True)
            self._check("type is message", r.get("type") == "message")
            self._check("has content", len(r.get("content", [])) > 0)
        except Exception as e:
            self._check("tool request Anthropic", False, str(e))

    # ── Sampling parameters ──────────────────────────────────────────────

    def test_temperature_zero(self):
        """Deterministic output at temperature=0."""
        print("\n[SP-1] Sampling — temperature=0 determinism")
        prompt = {
            "model": "dflash",
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            r1 = self._req("POST", "/v1/chat/completions", prompt)
            r2 = self._req("POST", "/v1/chat/completions", prompt)
            c1 = r1["choices"][0]["message"].get("content", "")
            c2 = r2["choices"][0]["message"].get("content", "")
            self._check("outputs match at temp=0", c1 == c2,
                         f"c1={c1[:50]!r} vs c2={c2[:50]!r}")
            print(f"    → output: {c1!r}")
        except Exception as e:
            self._check("temperature determinism", False, str(e))

    def test_max_tokens_limit(self):
        """Verify max_tokens is respected."""
        print("\n[SP-2] Sampling — max_tokens limit")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [
                    {"role": "user",
                     "content": "Write a very long essay about the history of mathematics."}
                ],
                "max_tokens": 50,
                "temperature": 0.0,
                "stream": False,
            })
            usage = r.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            self._check("completion_tokens <= 50", completion_tokens <= 50,
                         f"got {completion_tokens}")
            self._check("completion_tokens > 0", completion_tokens > 0)
            print(f"    → completion_tokens: {completion_tokens}")
        except Exception as e:
            self._check("max_tokens limit", False, str(e))

    def test_top_p_parameter(self):
        """Verify server accepts top_p parameter."""
        print("\n[SP-3] Sampling — top_p parameter")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 1024,
                "temperature": 0.8,
                "top_p": 0.9,
                "stream": False,
            })
            self._check("got response with top_p", True)
            content = r["choices"][0]["message"].get("content", "")
            self._check("content non-empty", len(content) > 0)
        except Exception as e:
            self._check("top_p parameter", False, str(e))

    # ── Concurrent requests ──────────────────────────────────────────────

    def test_concurrent_requests(self):
        """Multiple requests sent concurrently should all complete."""
        print("\n[CC-1] Concurrent requests (3 in parallel)")
        results = [None, None, None]
        errors = [None, None, None]

        def do_request(idx, prompt):
            try:
                r = self._req("POST", "/v1/chat/completions", {
                    "model": "dflash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.0,
                    "stream": False,
                }, timeout=180.0)
                results[idx] = r
            except Exception as e:
                errors[idx] = str(e)

        prompts = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?",
        ]

        t0 = time.monotonic()
        threads = []
        for i, p in enumerate(prompts):
            t = threading.Thread(target=do_request, args=(i, p))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=300)
        elapsed = time.monotonic() - t0

        # Note: dflash_server has a single worker thread, so requests
        # are serialized. But all should still complete.
        for i in range(3):
            if errors[i]:
                self._check(f"request {i+1} completes", False, errors[i])
            elif results[i] is None:
                self._check(f"request {i+1} completes", False, "timeout")
            else:
                content = results[i]["choices"][0]["message"].get("content", "")
                self._check(f"request {i+1} has content", len(content) > 0,
                             f"empty content")

        print(f"    → all 3 completed in {elapsed:.1f}s")

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_empty_user_message(self):
        """Empty user message should still get a response."""
        print("\n[EC-1] Edge case — empty user message")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [{"role": "user", "content": ""}],
                "max_tokens": 512,
                "temperature": 0.0,
                "stream": False,
            })
            self._check("got response", True)
            self._check("has choices", len(r.get("choices", [])) > 0)
        except Exception as e:
            self._check("empty user message", False, str(e))

    def test_long_system_prompt(self):
        """Long system prompt should work without truncation errors."""
        print("\n[EC-2] Edge case — long system prompt (2000 chars)")
        long_system = "You are a helpful assistant. " * 100  # ~2900 chars
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [
                    {"role": "system", "content": long_system},
                    {"role": "user", "content": "Say OK."}
                ],
                "max_tokens": 512,
                "temperature": 0.0,
                "stream": False,
            })
            self._check("got response with long system", True)
            content = r["choices"][0]["message"].get("content", "")
            self._check("content non-empty", len(content) > 0)
        except Exception as e:
            self._check("long system prompt", False, str(e))

    def test_unicode_content(self):
        """Unicode content in request and response."""
        print("\n[EC-3] Edge case — Unicode content")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [
                    {"role": "user", "content": "Translate 'hello' to Japanese (こんにちは)."}
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": False,
            })
            self._check("got response", True)
            content = r["choices"][0]["message"].get("content", "")
            self._check("content non-empty", len(content) > 0)
            # Verify the response is valid UTF-8 by checking encoding
            content.encode("utf-8")
            self._check("valid UTF-8 response", True)
            print(f"    → content: {content[:80]!r}")
        except UnicodeEncodeError:
            self._check("valid UTF-8 response", False, "invalid UTF-8")
        except Exception as e:
            self._check("unicode content", False, str(e))

    def test_multipart_content(self):
        """Array-style content (multi-part) should be handled."""
        print("\n[EC-4] Edge case — multi-part content array")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
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
            self._check("got response", True)
            content = r["choices"][0]["message"].get("content", "")
            self._check("content non-empty", len(content) > 0)
            print(f"    → content: {content!r}")
        except Exception as e:
            self._check("multi-part content", False, str(e))

    def test_invalid_json_body(self):
        """Completely invalid JSON body should return 400."""
        print("\n[EC-5] Edge case — invalid JSON body")
        try:
            url = self.base + "/v1/chat/completions"
            req = urllib.request.Request(url,
                data=b"not json at all{{{",
                headers={"Content-Type": "application/json"},
                method="POST")
            resp = urllib.request.urlopen(req, timeout=10)
            self._check("should have returned 400", False,
                         f"got {resp.status}")
        except urllib.error.HTTPError as e:
            self._check("returns 400 for invalid JSON",
                         e.code == 400, f"got {e.code}")
        except Exception as e:
            self._check("invalid JSON body", False, str(e))

    def test_options_cors(self):
        """OPTIONS request should return CORS headers."""
        print("\n[EC-6] Edge case — CORS OPTIONS preflight")
        try:
            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                method="OPTIONS"
            )
            resp = urllib.request.urlopen(req, timeout=10)
            headers = dict(resp.headers)
            self._check("returns 200",
                         resp.status == 200 or resp.status == 204)
            self._check("has Access-Control-Allow-Origin",
                         "Access-Control-Allow-Origin" in headers,
                         f"headers: {list(headers.keys())}")
        except Exception as e:
            self._check("CORS OPTIONS", False, str(e))

    # ── DDTree startup validation ────────────────────────────────────────

    def test_ddtree_flags_accepted(self):
        """Verify the server binary accepts --ddtree flags without crashing.
        (Actual DDTree requires Qwen35 backend — this just validates CLI parsing.)"""
        print("\n[DD-1] DDTree — CLI flag validation")
        try:
            binary = os.path.join(os.path.dirname(__file__),
                                  "..", "build", "dflash_server")
            if not os.path.exists(binary):
                self._skip("DDTree CLI validation", "binary not found")
                return
            # Run with --help or just check that --ddtree doesn't crash on parse.
            # We can't actually start with DDTree on Qwen3, but we can check
            # the binary prints usage correctly.
            proc = subprocess.run(
                [binary],
                capture_output=True, text=True, timeout=5
            )
            usage = proc.stderr + proc.stdout
            self._check("binary shows usage", "ddtree" in usage.lower() or
                         "--ddtree" in usage,
                         f"usage output: {usage[:200]}")
        except subprocess.TimeoutExpired:
            self._check("binary doesn't hang without args", False, "timed out")
        except Exception as e:
            self._check("DDTree CLI validation", False, str(e))

    # ── PFlash CLI validation ────────────────────────────────────────────

    def test_pflash_flags_accepted(self):
        """Verify the server binary shows pflash flags in usage."""
        print("\n[PF-1] PFlash — CLI flag validation")
        try:
            binary = os.path.join(os.path.dirname(__file__),
                                  "..", "build", "dflash_server")
            if not os.path.exists(binary):
                self._skip("PFlash CLI validation", "binary not found")
                return
            proc = subprocess.run(
                [binary],
                capture_output=True, text=True, timeout=5
            )
            usage = proc.stderr + proc.stdout
            self._check("usage shows --prefill-compression",
                         "--prefill-compression" in usage,
                         f"usage output: {usage[:300]}")
            self._check("usage shows --prefill-threshold",
                         "--prefill-threshold" in usage)
            self._check("usage shows --prefill-drafter",
                         "--prefill-drafter" in usage)
            self._check("usage shows --prefill-skip-park",
                         "--prefill-skip-park" in usage)
            self._check("usage shows --prefill-keep-ratio",
                         "--prefill-keep-ratio" in usage)
        except subprocess.TimeoutExpired:
            self._check("binary doesn't hang", False, "timed out")
        except Exception as e:
            self._check("PFlash CLI validation", False, str(e))

    def test_pflash_requires_drafter(self):
        """Server should fail with error when --prefill-compression enabled
        but --prefill-drafter not provided."""
        print("\n[PF-2] PFlash — requires --prefill-drafter")
        try:
            binary = os.path.join(os.path.dirname(__file__),
                                  "..", "build", "dflash_server")
            if not os.path.exists(binary):
                self._skip("PFlash drafter check", "binary not found")
                return
            # Give a model path + pflash flags but NO --prefill-drafter
            model = os.path.join(os.path.dirname(__file__),
                                 "..", "models", "Qwen3-0.6B-BF16.gguf")
            proc = subprocess.run(
                [binary, model, "--prefill-compression", "auto",
                 "--port", "19999"],
                capture_output=True, text=True, timeout=10
            )
            output = proc.stderr + proc.stdout
            self._check("server rejects missing drafter",
                         proc.returncode != 0 and "prefill-drafter" in output.lower(),
                         f"rc={proc.returncode} output: {output[:200]}")
        except subprocess.TimeoutExpired:
            self._check("server exits quickly on bad config", False, "timed out")
        except Exception as e:
            self._check("PFlash drafter check", False, str(e))

    # ── Streaming with client disconnect simulation ──────────────────────

    def test_streaming_partial_read(self):
        """Read only first few chunks of a streaming response, then close.
        Server should handle the disconnect gracefully."""
        print("\n[DC-1] Disconnect — partial stream read")
        try:
            resp = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [
                    {"role": "user",
                     "content": "Write a long story about a dragon."}
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": True,
            }, stream=True)

            # Read just a few lines then close.
            chunks_read = 0
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks_read += 1
                    if chunks_read >= 3:
                        break
            resp.close()

            self._check("read partial stream OK", chunks_read >= 1,
                         f"got {chunks_read} chunks")

            # Give server a moment to detect disconnect, then verify it's
            # still healthy.
            time.sleep(1)
            health = self._req("GET", "/health")
            self._check("server still healthy after disconnect",
                         health.get("status") == "ok")
        except Exception as e:
            self._check("partial stream disconnect", False, str(e))

    # ── Request ID format ────────────────────────────────────────────────

    def test_request_id_formats(self):
        """Verify each API format generates the correct ID prefix."""
        print("\n[ID-1] Request IDs — format-specific prefixes")
        try:
            r1 = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64, "temperature": 0.0, "stream": False,
            })
            self._check("OpenAI id starts with chatcmpl",
                         r1.get("id", "").startswith("chatcmpl"))

            r2 = self._req("POST", "/v1/messages", {
                "model": "dflash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64, "temperature": 0.0, "stream": False,
            })
            self._check("Anthropic id starts with msg",
                         r2.get("id", "").startswith("msg"))

            r3 = self._req("POST", "/v1/responses", {
                "model": "dflash",
                "input": "hi",
                "max_tokens": 64, "temperature": 0.0, "stream": False,
            })
            self._check("Responses id starts with resp",
                         r3.get("id", "").startswith("resp"))

        except Exception as e:
            self._check("request ID formats", False, str(e))

    # ── Run all ──────────────────────────────────────────────────────────

    def run_all(self):
        print("=" * 60)
        print("Comprehensive Server Tests")
        print(f"Target: {self.base}")
        print("=" * 60)

        # Prefix cache
        self.test_prefix_cache_timing()
        self.test_prefix_cache_shared_system()

        # Multi-turn
        self.test_multi_turn_openai()
        self.test_multi_turn_anthropic()
        self.test_multi_turn_responses()

        # Reasoning
        self.test_reasoning_nonstreaming_openai()
        self.test_reasoning_nonstreaming_anthropic()
        self.test_reasoning_streaming_openai()

        # Non-streaming all APIs
        self.test_nonstreaming_anthropic_full()
        self.test_nonstreaming_responses_full()
        self.test_nonstreaming_responses_string_input()

        # Streaming all APIs
        self.test_streaming_anthropic()
        self.test_streaming_responses()

        # Tools
        self.test_tool_request_format()
        self.test_tool_request_anthropic()

        # Sampling
        self.test_temperature_zero()
        self.test_max_tokens_limit()
        self.test_top_p_parameter()

        # Concurrent
        self.test_concurrent_requests()

        # Edge cases
        self.test_empty_user_message()
        self.test_long_system_prompt()
        self.test_unicode_content()
        self.test_multipart_content()
        self.test_invalid_json_body()
        self.test_options_cors()

        # DDTree
        self.test_ddtree_flags_accepted()

        # PFlash
        self.test_pflash_flags_accepted()
        self.test_pflash_requires_drafter()

        # Disconnect
        self.test_streaming_partial_read()

        # Request IDs
        self.test_request_id_formats()

        print("\n" + "=" * 60)
        total = self.passed + self.failed + self.skipped
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed"
              + (f", {self.skipped} skipped" if self.skipped else ""))
        return 0 if self.failed == 0 else 1


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive dflash_server tests")
    parser.add_argument("--base-url", default="http://localhost:9099",
                        help="Server base URL")
    parser.add_argument("--launch", metavar="MODEL",
                        help="Auto-launch server with given model path")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--server-log", help="Path to server log file for inspection")
    args = parser.parse_args()

    server_proc = None
    log_path = args.server_log

    if args.launch:
        log_path = log_path or "/tmp/dflash_server_test.log"
        binary = os.path.join(os.path.dirname(__file__), "..", "build", "dflash_server")
        cmd = [binary, args.launch, "--port", str(args.port)]
        with open(log_path, "w") as log_f:
            server_proc = subprocess.Popen(cmd, stderr=log_f, stdout=log_f)
        print(f"Launched server (PID {server_proc.pid}), waiting for startup...")
        base_url = f"http://localhost:{args.port}"
        for i in range(30):
            try:
                urllib.request.urlopen(base_url + "/health", timeout=2)
                print("Server ready.")
                break
            except Exception:
                time.sleep(1)
        else:
            print("ERROR: Server didn't start in 30s")
            server_proc.terminate()
            sys.exit(1)
    else:
        base_url = args.base_url

    try:
        suite = TestSuite(base_url, log_path)
        rc = suite.run_all()
    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait(timeout=10)
            print(f"Server stopped (log: {log_path})")

    sys.exit(rc)


if __name__ == "__main__":
    main()
