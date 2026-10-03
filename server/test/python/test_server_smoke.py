"""Smoke test for luce_server — exercises health, models, and generation endpoints.

Usage:
  # Against a running server:
  pytest server/test/python/test_server_smoke.py -v --base-url http://localhost:8080

  # Or let pytest launch one for the session:
  pytest server/test/python/test_server_smoke.py -v --launch <model.gguf>
"""

import json
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.server


class TestSmoke:
    @pytest.fixture(autouse=True)
    def _client(self, server_handle):
        self.client = server_handle.client

    def test_health(self):
        r = self.client.get("/health")
        assert r.get("status") == "ok", f"got: {r}"

    def test_root_health(self):
        r = self.client.get("/")
        assert r.get("status") == "ok", f"got: {r}"

    def test_models(self):
        r = self.client.get("/v1/models")
        assert r.get("object") == "list", f"got: {r.get('object')}"
        data = r.get("data", [])
        assert len(data) >= 1, f"got {len(data)} models"
        assert "id" in data[0], f"first model: {data[0]}"

    def test_chat_completion_streaming(self):
        resp = self.client.post_stream("/v1/chat/completions", {
            "model": "luce",
            "messages": [
                {"role": "user", "content": "Say hello in one word."}
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": True,
        }, timeout=120.0)

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

        assert chunks, "no chunks received"
        assert has_done, "no [DONE] sentinel"
        assert text, "accumulated text is empty"
        assert "id" in chunks[0]
        print(f"    → generated: {text!r}")

    def test_chat_completion_nonstreaming(self):
        r = self.client.post("/v1/chat/completions", {
            "model": "luce",
            "messages": [
                {"role": "user", "content": "What is 2+2? Reply with just the number."}
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False,
        }, timeout=120.0)

        assert r.get("object") == "chat.completion", f"got: {r.get('object')}"
        choices = r.get("choices", [])
        assert len(choices) >= 1
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        assert content, "message content is empty"
        assert choices[0].get("finish_reason") is not None
        usage = r.get("usage", {})
        assert "prompt_tokens" in usage, f"usage: {usage}"
        assert "completion_tokens" in usage
        print(f"    → content: {content!r}")

    def test_anthropic_messages(self):
        resp = self.client.post_stream("/v1/messages", {
            "model": "luce",
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Say hi in one word."}
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": True,
        }, timeout=120.0)

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

        assert events, "no events received"
        assert has_stop, "no message_stop event"
        assert text, "accumulated text is empty"
        assert any(e.get("type") == "message_start" for e in events)
        print(f"    → generated: {text!r}")

    def test_responses_api(self):
        resp = self.client.post_stream("/v1/responses", {
            "model": "luce",
            "input": "What is 1+1? Reply with just the number.",
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": True,
        }, timeout=120.0)

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

        assert events, "no events received"
        assert has_completed, "no response.completed event"
        assert text, "accumulated text is empty"
        assert any(e.get("type") == "response.created" for e in events)
        print(f"    → generated: {text!r}")

    def test_404(self):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            self.client.post("/v1/nonexistent", {"foo": "bar"})
        assert exc_info.value.code == 404

    def test_bad_json(self):
        req = urllib.request.Request(
            self.client.base_url + "/v1/chat/completions",
            data=b"not json at all",
            headers={"Content-Type": "application/json"},
            method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 400

