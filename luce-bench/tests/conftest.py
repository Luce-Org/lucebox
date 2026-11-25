"""Shared pytest fixtures.

Currently just the in-process mock OpenAI server (`mock_openai_server`).
Spins up a stdlib ``http.server.HTTPServer`` on a random localhost port,
serves canned ``/v1/models`` and ``/v1/chat/completions`` responses, and
yields the base URL. Used by ``tests/test_smoke_end_to_end.py`` to drive
the ``lucebench`` CLI against a real socket-level server without
needing an external service.

The server is intentionally minimal — enough to satisfy ``run_case`` /
``resolve_model`` / the sweep dispatch, not a full OpenAI surface.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Handler(BaseHTTPRequestHandler):
    """Minimal OpenAI-shape responder. Captures requests for assertions."""

    # Filled in by the fixture before the server starts.
    response_for: Callable[[dict], dict]
    captured: list[dict[str, Any]] = []

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet logs
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_from_completion(self, payload: dict) -> None:
        """Emit a streaming SSE response synthesized from a non-streaming
        OpenAI completion dict.

        Split the assistant content into 3 deltas so tests can observe a
        meaningful TTFT (first chunk arrives before the rest). The final
        chunk carries the usage block (matches OpenAI's behavior when
        stream_options.include_usage is true).
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        choices = payload.get("choices") or []
        choice0 = choices[0] if choices else {}
        msg = choice0.get("message", {}) if isinstance(choice0, dict) else {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish_reason = choice0.get("finish_reason") or "stop"
        model = payload.get("model") or "mock-model"

        def _send(frame: dict) -> None:
            line = "data: " + json.dumps(frame) + "\n\n"
            self.wfile.write(line.encode())
            self.wfile.flush()

        # Chunk 0: role-only opener (no content yet). TTFT should NOT
        # stamp here — the runner only stamps on chunks with real text.
        _send(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        # Split content into 3 roughly-equal pieces; reasoning into 1.
        if reasoning:
            _send(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": reasoning},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        if content:
            n = max(1, len(content) // 3)
            pieces = [content[i : i + n] for i in range(0, len(content), n)]
            for p in pieces:
                _send(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {"content": p}, "finish_reason": None}
                        ],
                    }
                )

        # Final chunk with finish_reason + usage (when client asked).
        final: dict[str, Any] = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        if isinstance(payload.get("usage"), dict):
            final["usage"] = payload["usage"]
        _send(final)
        # Terminator.
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._send_json({"object": "list", "data": [{"id": "mock-model", "object": "model"}]})
            return
        self._send_json({"error": {"message": "not found"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body_bytes = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(body_bytes)
        except ValueError:
            body = {}
        record = {
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
        }
        type(self).captured.append(record)
        if self.path == "/v1/chat/completions":
            payload = type(self).response_for(body)
            # Honor the request's stream flag — when the runner asks for
            # SSE we synthesize chunks from the same canned completion.
            if body.get("stream"):
                self._send_sse_from_completion(payload)
            else:
                self._send_json(payload)
            return
        self._send_json({"error": {"message": "not found"}}, status=404)


def _default_response(body: dict) -> dict:
    """The canned response used by default — echoes content '42\\nAnswer: 42'."""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model") or "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "42\nAnswer: 42"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "timings": {
                "prefill_ms": 5.0,
                "decode_ms": 50.0,
                "decode_tokens_per_sec": 100.0,
            },
        },
    }


@pytest.fixture
def mock_openai_server():
    """Yield a (base_url, captured_requests, set_response) triple.

    Example:
        def test_x(mock_openai_server):
            url, captured, set_response = mock_openai_server
            # ... drive lucebench against url ...
            assert captured[0]["body"]["model"] == "mock-model"
    """
    port = _free_port()

    class HandlerForThisTest(_Handler):
        pass

    HandlerForThisTest.response_for = staticmethod(_default_response)
    HandlerForThisTest.captured = []

    server = HTTPServer(("127.0.0.1", port), HandlerForThisTest)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def set_response(fn: Callable[[dict], dict]) -> None:
        HandlerForThisTest.response_for = staticmethod(fn)

    try:
        yield f"http://127.0.0.1:{port}", HandlerForThisTest.captured, set_response
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
