"""End-to-end integration test for the dflash HttpServer, driven by a
deterministic stub backend (no GPU, no model weights).

Runs the replay_http_server binary with:
  - the tokenizer-only Qwen3.6 GGUF fixture under server/test/fixtures/
  - the JSON scenario files under server/test/scenarios/

Then sends real HTTP requests at it via the same `requests` calls that
test_server_integration.py uses against a live GPU server. Because the
chat-template renderer, request parser, SseEmitter, and SSE socket
writes are the production code path, this test exercises the exact wire
that broke in the original Qwen3.6 think-channel bug — but on a CPU-only
runner with no model file.

Run locally:
  ./server/build/replay_http_server (built by cmake)
  uv run pytest server/test/test_stub_integration.py -v

The test starts and stops the driver itself; no separate server required.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests


REPO_ROOT     = Path(__file__).resolve().parents[2]
BUILD_DIR     = REPO_ROOT / "server" / "build"
DRIVER_BIN    = BUILD_DIR / "replay_http_server"
TOKENIZER_GGUF = REPO_ROOT / "server" / "test" / "fixtures" / "qwen3.6-tokenizer.gguf"
SCENARIOS_DIR = REPO_ROOT / "server" / "test" / "scenarios"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def stub_server():
    """Spawn the stub-driven HttpServer for the duration of this module."""
    assert DRIVER_BIN.is_file(), (
        f"driver binary missing: {DRIVER_BIN} — "
        "build target replay_http_server first")
    assert TOKENIZER_GGUF.is_file(), (
        f"tokenizer fixture missing: {TOKENIZER_GGUF} — "
        "is git-lfs configured for *.gguf?")

    port = _free_port()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.Popen(
        [str(DRIVER_BIN),
         str(TOKENIZER_GGUF),
         "--scenarios", str(SCENARIOS_DIR),
         "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"

    # Wait up to 10s for /health to come up.
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/health", timeout=1)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"driver exited early. output:\n{out}")
        time.sleep(0.1)
    else:
        proc.terminate()
        out = proc.stdout.read().decode() if proc.stdout else ""
        raise RuntimeError(f"driver did not become healthy in 10s. output:\n{out}")

    yield base

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


# ─── Tests ─────────────────────────────────────────────────────────────


class TestQwen3EnableThinkingNonStreaming:
    """Regression guard for the original PR #308 bug: Qwen3.6 enable_thinking
    must route reasoning into reasoning_content, not leak it into content."""

    def test_openai_chat_completions(self, stub_server):
        r = requests.post(f"{stub_server}/v1/chat/completions",
            json={
                "model": "dflash",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "chat_template_kwargs": {"enable_thinking": True},
                "stream": False,
                "max_tokens": 256,
            },
            timeout=10)
        assert r.status_code == 200, r.text
        msg = r.json()["choices"][0]["message"]
        reasoning = msg.get("reasoning_content") or ""
        content   = msg.get("content") or ""
        assert reasoning, (
            f"reasoning_content empty — render→emit wiring broken. "
            f"content={content!r}")
        assert "Let me compute" in reasoning
        assert "<think>"  not in reasoning
        assert "</think>" not in reasoning
        assert "<think>"  not in content
        assert "</think>" not in content
        assert "The answer is 4." in content
        assert "Let me compute" not in content, (
            f"reasoning text leaked into content: {content!r}")

    def test_anthropic_messages(self, stub_server):
        r = requests.post(f"{stub_server}/v1/messages",
            json={
                "model": "dflash",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "thinking": {"type": "enabled"},
                "stream": False,
                "max_tokens": 256,
            },
            timeout=10)
        assert r.status_code == 200, r.text
        blocks = r.json().get("content", [])
        types = [b.get("type") for b in blocks]
        assert "thinking" in types, f"no thinking block; types={types}"
        assert "text" in types,     f"no text block; types={types}"
        thinking = next(b["thinking"] for b in blocks if b["type"] == "thinking")
        text     = next(b["text"]     for b in blocks if b["type"] == "text")
        assert "Let me compute"  in thinking
        assert "The answer is 4" in text
        assert "<think>"  not in thinking
        assert "</think>" not in thinking


class TestQwen3EnableThinkingStreaming:
    """Same bug class but through the streaming SSE code path."""

    def test_openai_chat_completions_streaming(self, stub_server):
        r = requests.post(f"{stub_server}/v1/chat/completions",
            json={
                "model": "dflash",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "chat_template_kwargs": {"enable_thinking": True},
                "stream": True,
                "max_tokens": 256,
            },
            stream=True, timeout=10)
        assert r.status_code == 200

        reasoning_text = ""
        content_text = ""
        saw_reasoning_delta = False
        saw_content_delta = False
        saw_done = False
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line == "data: [DONE]":
                saw_done = True
                break
            assert line.startswith("data: "), line
            chunk = json.loads(line[len("data: "):])
            choices = chunk.get("choices") or []
            if not choices:
                continue   # usage-only tail chunk; no delta
            delta = choices[0].get("delta", {})
            if "reasoning_content" in delta:
                saw_reasoning_delta = True
                reasoning_text += delta["reasoning_content"]
            if "content" in delta:
                saw_content_delta = True
                content_text += delta["content"]
        assert saw_done
        assert saw_reasoning_delta, "no reasoning_content deltas emitted"
        assert saw_content_delta,   "no content deltas emitted"
        assert "Let me compute"  in reasoning_text
        assert "The answer is 4" in content_text
        assert "<think>"  not in reasoning_text
        assert "</think>" not in reasoning_text
        assert "<think>"  not in content_text
        assert "</think>" not in content_text

    def test_anthropic_messages_streaming(self, stub_server):
        """First content_block_start must be `thinking`, not `text` — the
        Anthropic-side half of the PR #308 fix."""
        r = requests.post(f"{stub_server}/v1/messages",
            json={
                "model": "dflash",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "thinking": {"type": "enabled"},
                "stream": True,
                "max_tokens": 256,
            },
            stream=True, timeout=10)
        assert r.status_code == 200

        events = []
        event_type = None
        for line in r.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: ") and event_type:
                events.append((event_type, json.loads(line[len("data: "):])))
                event_type = None
        types = [t for t, _ in events]
        assert "message_start"      in types
        first_start = next(d for t, d in events if t == "content_block_start")
        assert first_start["content_block"]["type"] == "thinking", first_start
        # At least one thinking delta and one text delta.
        thinking_deltas = [d for t, d in events
                           if t == "content_block_delta"
                           and d["delta"]["type"] == "thinking_delta"]
        text_deltas = [d for t, d in events
                       if t == "content_block_delta"
                       and d["delta"]["type"] == "text_delta"]
        assert thinking_deltas, f"no thinking_delta events; types={types}"
        assert text_deltas,     f"no text_delta events; types={types}"
        assert "Let me compute" in "".join(d["delta"]["thinking"] for d in thinking_deltas)
        assert "The answer is 4" in "".join(d["delta"]["text"] for d in text_deltas)
