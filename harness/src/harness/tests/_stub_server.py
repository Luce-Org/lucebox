"""Minimal ThreadingHTTPServer request recorder for harness tests.

Matches the pattern already used in harness/clients/llamacpp_compat_proxy.py
(http.server.ThreadingHTTPServer, stdlib-only, no new deps).

Usage:
    with StubServer() as stub:
        # stub.url -> "http://127.0.0.1:<port>"
        # make requests here
        req = stub.last_request()  # dict with path, method, headers, body
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _RecordingHandler(BaseHTTPRequestHandler):
    """Records every request; replies with a minimal valid fixture response."""

    def log_message(self, fmt, *args):  # silence default stderr logging
        pass

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(n) if n > 0 else b""

    def _reply_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self) -> None:
        body = self._read_body()
        record: dict[str, Any] = {
            "method": self.command,
            "path": self.path,
            "headers": dict(self.headers),
            "body_bytes": body,
            "body_json": None,
        }
        try:
            record["body_json"] = json.loads(body.decode("utf-8")) if body else None
        except Exception:
            pass
        self.server._requests.append(record)  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        self._record()
        if self.path.startswith("/health"):
            self._reply_json({"status": "ok"})
        elif self.path.startswith("/v1/models"):
            self._reply_json({"object": "list", "data": [{"id": "luce-dflash"}]})
        else:
            self._reply_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        self._record()
        if self.path.startswith("/v1/messages"):
            self._reply_json({
                "id": "stub-msg-1",
                "type": "message",
                "role": "assistant",
                "model": "luce-dflash",
                "content": [{"type": "text", "text": "lucebox stub response"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            })
        elif self.path.startswith("/v1/chat/completions"):
            self._reply_json({
                "id": "stub-chat-1",
                "object": "chat.completion",
                "model": "luce-dflash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "lucebox stub response"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            })
        elif self.path.startswith("/v1/responses"):
            self._reply_json({
                "id": "stub-resp-1",
                "object": "response",
                "model": "luce-dflash",
                "output_text": "lucebox stub response",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "lucebox stub response"}],
                }],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            })
        else:
            self._reply_json({"error": "not found"}, 404)


class StubServer:
    """Context manager wrapping a ThreadingHTTPServer on a random local port."""

    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url: str = ""

    def __enter__(self) -> "StubServer":
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
        srv._requests: list[dict[str, Any]] = []  # type: ignore[attr-defined]
        self._server = srv
        port = srv.server_address[1]
        self.url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        if self._server:
            self._server.shutdown()

    def requests(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded requests."""
        return list(self._server._requests)  # type: ignore[union-attr]

    def last_request(self) -> dict[str, Any] | None:
        reqs = self.requests()
        return reqs[-1] if reqs else None

    def clear(self) -> None:
        if self._server:
            self._server._requests.clear()  # type: ignore[attr-defined]
