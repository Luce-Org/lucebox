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
            self._send_json(type(self).response_for(body))
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
