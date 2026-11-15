#!/usr/bin/env python3
"""Regression tests for client harness installation and probe reporting."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from harness import client_test_runner as runner


class OmpInstallTests(unittest.TestCase):
    def test_download_failure_returns_structured_install_result(self) -> None:
        spec = runner.CLIENTS["omp"]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.object(
                    runner.urllib.request,
                    "urlopen",
                    side_effect=urllib.error.URLError("offline"),
                ) as urlopen,
                mock.patch.object(runner, "run_cmd") as run_cmd,
            ):
                result = runner.install_client(Path(temp_dir), spec)

        urlopen.assert_called_once_with(spec.package, timeout=60)
        run_cmd.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["cmd"], ["download", spec.package])
        self.assertIn("offline", result["output_tail"])
        self.assertEqual(result["client"], "omp")
        self.assertEqual(result["installer"], "omp")
        self.assertFalse(result["binary_exists"])
        self.assertFalse(result["version"]["ok"])


class _FakeStream(io.BytesIO):
    """Stands in for the urlopen() response of a streaming request."""

    status = 200

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _sse(*frames: tuple[str | None, object]) -> bytes:
    """Encode (event, data) frames the way luce_server's SseEmitter does."""
    out = []
    for event, data in frames:
        payload = data if isinstance(data, str) else json.dumps(data)
        prefix = f"event: {event}\n" if event else ""
        out.append(f"{prefix}data: {payload}\n\n")
    return "".join(out).encode("utf-8")


def _run_stream_probe(probe, body: bytes) -> dict:
    with mock.patch.object(
        runner.urllib.request, "urlopen", return_value=_FakeStream(body)
    ):
        return probe()


CHAT_DELTA = (None, {"choices": [{"delta": {"content": "lucebox par"}}]})
CHAT_FINISH = (None, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
CHAT_ERROR = (None, {"error": {"message": "backend died", "type": "server_error"}})
CHAT_DONE = (None, "[DONE]")

ANTHROPIC_START = ("message_start", {"type": "message_start"})
ANTHROPIC_DELTA = ("content_block_delta", {
    "type": "content_block_delta",
    "delta": {"type": "text_delta", "text": "lucebox par"},
})
ANTHROPIC_STOP = ("message_stop", {"type": "message_stop"})
ANTHROPIC_ERROR = ("error", {
    "type": "error",
    "error": {"type": "api_error", "message": "backend died"},
})

RESPONSES_CREATED = ("response.created", {"type": "response.created"})
RESPONSES_DELTA = ("response.output_text.delta", {
    "type": "response.output_text.delta",
    "delta": "lucebox par",
})


def _responses_terminal(event: str, status: str) -> tuple[str, dict]:
    return (event, {"type": event, "response": {"status": status}})


class ChatStreamProbeTests(unittest.TestCase):
    def probe(self, *frames: tuple[str | None, object]) -> dict:
        return _run_stream_probe(
            lambda: runner.http_sse("http://test/v1/chat/completions", {}),
            _sse(*frames),
        )

    def test_non_object_chunks_are_ignored(self) -> None:
        result = self.probe((None, [1, 2]), (None, 7), CHAT_DELTA, CHAT_FINISH, CHAT_DONE)

        self.assertTrue(result["ok"])

    def test_null_error_field_is_not_an_error(self) -> None:
        chunk = (None, {"error": None, "choices": [{"delta": {"content": "lucebox"}}]})

        result = self.probe(chunk, CHAT_FINISH, CHAT_DONE)

        self.assertTrue(result["ok"])
        self.assertIsNone(result["stream_error"])

    def test_empty_error_object_fails(self) -> None:
        result = self.probe(CHAT_DELTA, (None, {"error": {}}), CHAT_DONE)

        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["stream_error"])

    def test_complete_stream_passes(self) -> None:
        result = self.probe(CHAT_DELTA, CHAT_FINISH, CHAT_DONE)

        self.assertTrue(result["ok"])
        self.assertTrue(result["stream_completed"])
        self.assertIsNone(result["stream_error"])

    def test_error_chunk_after_partial_text_fails(self) -> None:
        result = self.probe(CHAT_DELTA, CHAT_ERROR, CHAT_DONE)

        self.assertFalse(result["ok"])
        self.assertIn("backend died", result["stream_error"])

    def test_stream_without_done_marker_fails(self) -> None:
        result = self.probe(CHAT_DELTA)

        self.assertFalse(result["ok"])
        self.assertFalse(result["stream_completed"])


class AnthropicStreamProbeTests(unittest.TestCase):
    def probe(self, *frames: tuple[str | None, object]) -> dict:
        return self.probe_bytes(_sse(*frames))

    def probe_bytes(self, body: bytes) -> dict:
        return _run_stream_probe(
            lambda: runner.probe_anthropic_sse("http://test/v1/messages", {}),
            body,
        )

    def test_error_event_with_unparseable_data_fails(self) -> None:
        result = self.probe(
            ANTHROPIC_START, ANTHROPIC_DELTA, ("error", "not json"), ANTHROPIC_STOP
        )

        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["stream_error"])

    def test_stream_cut_after_message_stop_header_fails(self) -> None:
        body = _sse(ANTHROPIC_START, ANTHROPIC_DELTA) + b"event: message_stop\n"

        result = self.probe_bytes(body)

        self.assertFalse(result["ok"])
        self.assertFalse(result["stream_completed"])

    def test_complete_stream_passes(self) -> None:
        result = self.probe(ANTHROPIC_START, ANTHROPIC_DELTA, ANTHROPIC_STOP)

        self.assertTrue(result["ok"])
        self.assertTrue(result["stream_completed"])
        self.assertIsNone(result["stream_error"])

    def test_error_event_after_partial_text_fails(self) -> None:
        result = self.probe(ANTHROPIC_START, ANTHROPIC_DELTA, ANTHROPIC_ERROR)

        self.assertFalse(result["ok"])
        self.assertIn("backend died", result["stream_error"])

    def test_stream_without_message_stop_fails(self) -> None:
        result = self.probe(ANTHROPIC_START, ANTHROPIC_DELTA)

        self.assertFalse(result["ok"])
        self.assertFalse(result["stream_completed"])


class ResponsesStreamProbeTests(unittest.TestCase):
    def probe(self, *frames: tuple[str | None, object]) -> dict:
        return self.probe_bytes(_sse(*frames))

    def probe_bytes(self, body: bytes) -> dict:
        return _run_stream_probe(
            lambda: runner.probe_responses_sse("http://test/v1/responses", {}),
            body,
        )

    def test_stream_cut_after_completed_header_fails(self) -> None:
        body = _sse(RESPONSES_CREATED, RESPONSES_DELTA) + b"event: response.completed\n"

        result = self.probe_bytes(body)

        self.assertFalse(result["ok"])
        self.assertFalse(result["stream_completed"])

    def test_failed_event_with_unparseable_data_fails(self) -> None:
        result = self.probe(
            RESPONSES_CREATED, RESPONSES_DELTA, ("response.failed", "not json")
        )

        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["stream_error"])

    def test_completed_stream_passes(self) -> None:
        result = self.probe(
            RESPONSES_CREATED,
            RESPONSES_DELTA,
            _responses_terminal("response.completed", "completed"),
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["stream_completed"])
        self.assertIsNone(result["stream_error"])

    def test_incomplete_stream_passes(self) -> None:
        # The probe caps max_output_tokens, so a healthy server normally ends
        # the turn with response.incomplete rather than response.completed.
        result = self.probe(
            RESPONSES_CREATED,
            RESPONSES_DELTA,
            _responses_terminal("response.incomplete", "incomplete"),
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["stream_completed"])

    def test_failed_event_after_partial_text_fails(self) -> None:
        failed = ("response.failed", {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"code": "server_error", "message": "backend died"},
            },
        })

        result = self.probe(RESPONSES_CREATED, RESPONSES_DELTA, failed)

        self.assertFalse(result["ok"])
        self.assertIn("backend died", result["stream_error"])

    def test_stream_without_terminal_event_fails(self) -> None:
        result = self.probe(RESPONSES_CREATED, RESPONSES_DELTA)

        self.assertFalse(result["ok"])
        self.assertFalse(result["stream_completed"])


if __name__ == "__main__":
    unittest.main()
