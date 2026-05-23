"""Tests for session_inject_proxy.py.

Seed tests (in plan order):
  #2 - test_session_injector_anthropic_messages_round_trip  (regression lock)
  #1 - test_session_injector_openai_chat_completions_round_trip  (OpenAI injection route)
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

# Allow running from repo root or harness/tests directly.
HARNESS_DIR = Path(__file__).resolve().parent.parent
if str(HARNESS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR.parent))

from harness.tests._stub_server import StubServer
from harness.clients.session_inject_proxy import Handler, main as proxy_main
from http.server import ThreadingHTTPServer


def _start_proxy(upstream_url: str, session_id: str, host: str = "127.0.0.1") -> tuple[ThreadingHTTPServer, str]:
    """Start a session_inject_proxy pointing at upstream_url, return (srv, proxy_url)."""
    Handler.upstream = upstream_url.rstrip("/")
    Handler.session_id = session_id
    srv = ThreadingHTTPServer((host, 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    port = srv.server_address[1]
    return srv, f"http://{host}:{port}"


class TestSessionInjectorAnthropicMessages(unittest.TestCase):
    """Seed #2 — regression lock: proxy injects session_id on /v1/messages."""

    def test_session_injector_anthropic_messages_round_trip(self):
        """POST /v1/messages through proxy → upstream sees injected session_id."""
        with StubServer() as stub:
            proxy_srv, proxy_url = _start_proxy(stub.url, session_id="test-sess-001")
            try:
                payload = {
                    "model": "luce-dflash",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 16,
                }
                body = json.dumps(payload).encode()
                req = urllib.request.Request(
                    proxy_url + "/v1/messages",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status = resp.status
                    resp_body = json.loads(resp.read())
            finally:
                proxy_srv.shutdown()

        # Response routed correctly
        self.assertEqual(status, 200)
        self.assertEqual(resp_body.get("type"), "message")

        # Upstream received the injected session_id
        upstream_req = stub.last_request()
        self.assertIsNotNone(upstream_req)
        self.assertEqual(upstream_req["method"], "POST")
        self.assertEqual(upstream_req["path"], "/v1/messages")
        upstream_body = upstream_req["body_json"]
        self.assertIsNotNone(upstream_body)
        extra = upstream_body.get("extra_body", {})
        self.assertEqual(extra.get("session_id"), "test-sess-001")

    def test_session_injector_does_not_overwrite_existing_session_id(self):
        """If client already set extra_body.session_id, proxy must not overwrite it."""
        with StubServer() as stub:
            proxy_srv, proxy_url = _start_proxy(stub.url, session_id="proxy-sess")
            try:
                payload = {
                    "model": "luce-dflash",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8,
                    "extra_body": {"session_id": "client-sess"},
                }
                body = json.dumps(payload).encode()
                req = urllib.request.Request(
                    proxy_url + "/v1/messages",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
            finally:
                proxy_srv.shutdown()

        upstream_req = stub.last_request()
        upstream_body = upstream_req["body_json"]
        # Must preserve client's session_id, not overwrite with proxy's
        self.assertEqual(upstream_body["extra_body"]["session_id"], "client-sess")

    def test_session_injector_passthrough_on_unknown_path(self):
        """Unknown paths outside INJECT_ROUTES are forwarded verbatim."""
        with StubServer() as stub:
            proxy_srv, proxy_url = _start_proxy(stub.url, session_id="proxy-sess")
            try:
                # /health is a GET, not an inject route
                req = urllib.request.Request(
                    proxy_url + "/health",
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
            finally:
                proxy_srv.shutdown()

        upstream_req = stub.last_request()
        self.assertEqual(upstream_req["method"], "GET")
        self.assertEqual(upstream_req["path"], "/health")


class TestSessionInjectorOpenAIChatCompletions(unittest.TestCase):
    """Seed #1 — OpenAI /v1/chat/completions injection route."""

    def test_session_injector_openai_chat_completions_round_trip(self):
        """POST /v1/chat/completions through proxy → upstream sees injected session_id."""
        with StubServer() as stub:
            proxy_srv, proxy_url = _start_proxy(stub.url, session_id="oai-sess-001")
            try:
                payload = {
                    "model": "luce-dflash",
                    "messages": [{"role": "user", "content": "hello openai"}],
                    "max_tokens": 16,
                }
                body = json.dumps(payload).encode()
                req = urllib.request.Request(
                    proxy_url + "/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status = resp.status
                    resp_body = json.loads(resp.read())
            finally:
                proxy_srv.shutdown()

        self.assertEqual(status, 200)
        upstream_req = stub.last_request()
        self.assertIsNotNone(upstream_req)
        self.assertEqual(upstream_req["path"], "/v1/chat/completions")
        upstream_body = upstream_req["body_json"]
        # Injection must happen on chat/completions (INJECT_ROUTES)
        extra = upstream_body.get("extra_body", {})
        self.assertEqual(extra.get("session_id"), "oai-sess-001")

    def test_session_injector_responses_round_trip(self):
        """POST /v1/responses through proxy → upstream sees injected session_id."""
        with StubServer() as stub:
            proxy_srv, proxy_url = _start_proxy(stub.url, session_id="resp-sess-001")
            try:
                payload = {
                    "model": "luce-dflash",
                    "input": [{"type": "message", "role": "user",
                               "content": [{"type": "input_text", "text": "hello"}]}],
                    "max_output_tokens": 16,
                }
                body = json.dumps(payload).encode()
                req = urllib.request.Request(
                    proxy_url + "/v1/responses",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status = resp.status
                    resp.read()
            finally:
                proxy_srv.shutdown()

        self.assertEqual(status, 200)
        upstream_req = stub.last_request()
        upstream_body = upstream_req["body_json"]
        extra = upstream_body.get("extra_body", {})
        self.assertEqual(extra.get("session_id"), "resp-sess-001")


if __name__ == "__main__":
    unittest.main()
