#!/usr/bin/env python3
"""Smoke test for dflash_server — exercises health, models, and generation endpoints.

Usage:
  # Start the server first, then run:
  python3 dflash/tests/test_server_smoke.py [--base-url http://localhost:8080]

  # Or use --launch to start the server automatically:
  python3 dflash/tests/test_server_smoke.py --launch <model.gguf> [--port 9099]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error


class SmokeTest:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def _req(self, method: str, path: str, body: dict | None = None,
             stream: bool = False, timeout: float = 120.0):
        url = self.base + path
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if body else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        resp = urllib.request.urlopen(req, timeout=timeout)
        if stream:
            return resp  # caller reads lines
        return json.loads(resp.read().decode())

    def _report(self, name: str, ok: bool, detail: str = ""):
        if ok:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {detail}")

    # ── Tests ──────────────────────────────────────────────────────────

    def test_health(self):
        print("\n[1] Health endpoint")
        try:
            r = self._req("GET", "/health")
            self._report("GET /health returns 200", True)
            self._report("status == ok", r.get("status") == "ok",
                         f"got: {r}")
        except Exception as e:
            self._report("GET /health", False, str(e))

    def test_root_health(self):
        print("\n[2] Root endpoint (alias for health)")
        try:
            r = self._req("GET", "/")
            self._report("GET / returns 200", True)
            self._report("status == ok", r.get("status") == "ok",
                         f"got: {r}")
        except Exception as e:
            self._report("GET /", False, str(e))

    def test_models(self):
        print("\n[3] Models endpoint")
        try:
            r = self._req("GET", "/v1/models")
            self._report("GET /v1/models returns 200", True)
            self._report("object == list", r.get("object") == "list",
                         f"got: {r.get('object')}")
            data = r.get("data", [])
            self._report("data has >=1 model", len(data) >= 1,
                         f"got {len(data)} models")
            if data:
                self._report("model has id", "id" in data[0],
                             f"first model: {data[0]}")
        except Exception as e:
            self._report("GET /v1/models", False, str(e))

    def test_chat_completion_streaming(self):
        print("\n[4] OpenAI chat/completions (streaming)")
        try:
            resp = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [
                    {"role": "user", "content": "Say hello in one word."}
                ],
                "max_tokens": 256,
                "temperature": 0.0,
                "stream": True,
            }, stream=True, timeout=120.0)

            chunks = []
            text = ""
            has_done = False
            for line in resp:
                line = line.decode().strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    has_done = True
                    break
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    chunks.append(chunk)
                    choices = chunk.get("choices") or [{}]
                    delta = choices[0].get("delta", {})
                    if "content" in delta:
                        text += delta["content"]

            self._report("received SSE chunks", len(chunks) > 0,
                         "no chunks received")
            self._report("got [DONE] sentinel", has_done)
            self._report("accumulated text non-empty", len(text) > 0,
                         f"text='{text}'")
            self._report("first chunk has id", "id" in chunks[0] if chunks else False)
            print(f"    → generated: {text!r}")
        except Exception as e:
            self._report("streaming chat completion", False, str(e))

    def test_chat_completion_nonstreaming(self):
        print("\n[5] OpenAI chat/completions (non-streaming)")
        try:
            r = self._req("POST", "/v1/chat/completions", {
                "model": "dflash",
                "messages": [
                    {"role": "user", "content": "What is 2+2? Reply with just the number."}
                ],
                "max_tokens": 256,
                "temperature": 0.0,
                "stream": False,
            }, timeout=120.0)

            self._report("got response", True)
            self._report("object == chat.completion",
                         r.get("object") == "chat.completion",
                         f"got: {r.get('object')}")
            choices = r.get("choices", [])
            self._report("choices has >=1 entry", len(choices) >= 1)
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                self._report("message has content", len(content) > 0,
                             f"content='{content}'")
                self._report("has finish_reason",
                             choices[0].get("finish_reason") is not None)
                print(f"    → content: {content!r}")
            usage = r.get("usage", {})
            self._report("usage has prompt_tokens",
                         "prompt_tokens" in usage, f"usage: {usage}")
            self._report("usage has completion_tokens",
                         "completion_tokens" in usage)
        except Exception as e:
            self._report("non-streaming chat completion", False, str(e))

    def test_anthropic_messages(self):
        print("\n[6] Anthropic /v1/messages (streaming)")
        try:
            resp = self._req("POST", "/v1/messages", {
                "model": "dflash",
                "system": "You are a helpful assistant.",
                "messages": [
                    {"role": "user", "content": "Say hi in one word."}
                ],
                "max_tokens": 256,
                "temperature": 0.0,
                "stream": True,
            }, stream=True, timeout=120.0)

            events = []
            text = ""
            has_stop = False
            for line in resp:
                line = line.decode().strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    evt = json.loads(line[6:])
                    events.append(evt)
                    if evt.get("type") == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text += delta.get("text", "")
                    if evt.get("type") == "message_stop":
                        has_stop = True

            self._report("received events", len(events) > 0,
                         "no events received")
            self._report("got message_stop", has_stop)
            self._report("accumulated text non-empty", len(text) > 0,
                         f"text='{text}'")
            has_start = any(e.get("type") == "message_start" for e in events)
            self._report("has message_start event", has_start)
            print(f"    → generated: {text!r}")
        except Exception as e:
            self._report("anthropic messages", False, str(e))

    def test_responses_api(self):
        print("\n[7] Responses /v1/responses (streaming)")
        try:
            resp = self._req("POST", "/v1/responses", {
                "model": "dflash",
                "input": "What is 1+1? Reply with just the number.",
                "max_tokens": 256,
                "temperature": 0.0,
                "stream": True,
            }, stream=True, timeout=120.0)

            events = []
            text = ""
            has_completed = False
            for line in resp:
                line = line.decode().strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    evt = json.loads(line[6:])
                    events.append(evt)
                    if evt.get("type") == "response.output_text.delta":
                        text += evt.get("delta", "")
                    if evt.get("type") == "response.completed":
                        has_completed = True

            self._report("received events", len(events) > 0,
                         "no events received")
            self._report("got response.completed", has_completed)
            self._report("accumulated text non-empty", len(text) > 0,
                         f"text='{text}'")
            has_created = any(e.get("type") == "response.created" for e in events)
            self._report("has response.created event", has_created)
            print(f"    → generated: {text!r}")
        except Exception as e:
            self._report("responses API", False, str(e))

    def test_404(self):
        print("\n[8] Unknown endpoint returns error")
        try:
            self._req("POST", "/v1/nonexistent", {"foo": "bar"})
            self._report("returns 404", False, "expected 404 but got 200")
        except urllib.error.HTTPError as e:
            self._report("returns 404", e.code == 404,
                         f"got HTTP {e.code}")
        except Exception as e:
            self._report("unknown endpoint", False, str(e))

    def test_bad_json(self):
        print("\n[9] Bad JSON body returns 400")
        try:
            url = self.base + "/v1/chat/completions"
            req = urllib.request.Request(
                url, data=b"not json at all",
                headers={"Content-Type": "application/json"},
                method="POST")
            urllib.request.urlopen(req, timeout=10)
            self._report("returns 400", False, "expected 400 but got 200")
        except urllib.error.HTTPError as e:
            self._report("returns 400", e.code == 400,
                         f"got HTTP {e.code}")
        except Exception as e:
            self._report("bad JSON", False, str(e))

    # ── Runner ─────────────────────────────────────────────────────────

    def run_all(self):
        print(f"Smoke testing: {self.base}")

        # Non-generation tests first (fast)
        self.test_health()
        self.test_root_health()
        self.test_models()
        self.test_404()
        self.test_bad_json()

        # Generation tests (require model loaded)
        self.test_chat_completion_streaming()
        self.test_chat_completion_nonstreaming()
        self.test_anthropic_messages()
        self.test_responses_api()

        # Summary
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} passed, "
              f"{self.failed} failed, {self.skipped} skipped")
        return self.failed == 0


def wait_for_server(base_url: str, timeout: float = 120.0):
    """Wait for the server to become ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(f"{base_url}/health", timeout=5)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="dflash_server smoke test")
    parser.add_argument("--base-url", default="http://localhost:8080",
                        help="Server base URL")
    parser.add_argument("--launch", metavar="MODEL_PATH",
                        help="Launch dflash_server with this model before testing")
    parser.add_argument("--port", type=int, default=9099,
                        help="Port to use when --launch is specified")
    parser.add_argument("--server-bin", default=None,
                        help="Path to dflash_server binary")
    args = parser.parse_args()

    server_proc = None
    try:
        if args.launch:
            port = args.port
            base_url = f"http://localhost:{port}"

            # Find binary
            bin_path = args.server_bin
            if not bin_path:
                candidates = [
                    "dflash/build/dflash_server",
                    "build/dflash_server",
                ]
                for c in candidates:
                    if os.path.isfile(c):
                        bin_path = c
                        break
            if not bin_path:
                print("ERROR: Could not find dflash_server binary")
                sys.exit(1)

            print(f"Launching: {bin_path} {args.launch} --port {port}")
            server_proc = subprocess.Popen(
                [bin_path, args.launch, "--port", str(port),
                 "--max-ctx", "4096", "--max-tokens", "64"],
                stderr=subprocess.PIPE,
            )

            print("Waiting for server to start...")
            if not wait_for_server(base_url, timeout=120.0):
                print("ERROR: Server did not start within 120s")
                server_proc.terminate()
                stderr = server_proc.stderr.read().decode()
                print(f"Server stderr:\n{stderr}")
                sys.exit(1)
            print("Server ready!")
            args.base_url = base_url

        tester = SmokeTest(args.base_url)
        ok = tester.run_all()
        sys.exit(0 if ok else 1)

    finally:
        if server_proc:
            print("\nShutting down server...")
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            print("Server stopped.")


if __name__ == "__main__":
    main()
