from __future__ import annotations

import http.client
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import ds4_context_sweep  # noqa: E402
import ds4_publication_decode_client  # noqa: E402


class _StreamingResponse:
    status = 200

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> _StreamingResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


class PublicationClientTests(unittest.TestCase):
    def test_short_completion_is_a_failed_measurement(self) -> None:
        response = _StreamingResponse(
            [
                b'data: {"choices":[{"delta":{"content":"1\\n"}}]}\n',
                b'data: {"usage":{"prompt_tokens":12,"completion_tokens":1}}\n',
                b"data: [DONE]\n",
            ]
        )
        with mock.patch.object(
            ds4_publication_decode_client.urllib.request,
            "urlopen",
            return_value=response,
        ):
            result = ds4_publication_decode_client.stream_request(
                "http://127.0.0.1:1", "dflash", "prompt", max_tokens=2
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["completion_tokens"], 1)
        self.assertIn("short completion", result["error"])

    def test_incomplete_http_stream_is_reported(self) -> None:
        with mock.patch.object(
            ds4_publication_decode_client.urllib.request,
            "urlopen",
            side_effect=http.client.IncompleteRead(b"partial"),
        ):
            result = ds4_publication_decode_client.stream_request(
                "http://127.0.0.1:1", "dflash", "prompt", max_tokens=2
            )

        self.assertFalse(result["ok"])
        self.assertIn("IncompleteRead", result["error"])

    def test_zero_runs_is_rejected(self) -> None:
        argv = ["publication-client", "--json-out", "unused.json", "--runs", "0"]
        with (
            mock.patch.object(sys, "argv", argv),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            ds4_publication_decode_client.main()

        self.assertEqual(error.exception.code, 2)


class ContextSummaryTests(unittest.TestCase):
    def test_ok_row_without_decode_metrics_is_excluded(self) -> None:
        summary = ds4_context_sweep.summarize(
            [
                {"ok": True, "prompt_tokens": 2048, "completion_tokens": 128},
                {
                    "ok": True,
                    "prompt_tokens": 2048,
                    "completion_tokens": 128,
                    "client_decode_s": 2.0,
                    "client_decode_tok_s": 64.0,
                    "response_sha256": "abc",
                },
            ]
        )

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["n_ok"], 1)
        self.assertEqual(summary["client_decode_tok_s_weighted"], 64.0)


if __name__ == "__main__":
    unittest.main()
