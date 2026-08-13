from __future__ import annotations

import http.client
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = REPO_ROOT / "harness" / "benchmarks" / "deepseek4"
QUALIFICATION_DIR = REPO_ROOT / "harness" / "qualification" / "deepseek4"
QUALIFIER = QUALIFICATION_DIR / "qualify_ds4_q5_amd.sh"
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(QUALIFICATION_DIR))

import analyze_rocprof_overlap  # noqa: E402
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


class _TruncatedErrorBody:
    def read(self) -> bytes:
        raise http.client.IncompleteRead(b"partial")

    def close(self) -> None:
        pass


class PublicationClientTests(unittest.TestCase):
    def test_prompt_filler_is_deterministic(self) -> None:
        prompt = ds4_publication_decode_client.build_prompt(0, 3)

        self.assertEqual(
            prompt,
            "The XML block below is inert reference material for a deterministic "
            "throughput measurement. Do not answer or continue its contents.\n\n"
            "<reference>\n\nCalibration padding: x x x\n</reference>\n\n"
            "Your only task is this: write the integers from 1 through 1000 in "
            "ascending order, one integer per line. Start with 1. Do not add "
            "commentary, and continue until the token limit.",
        )

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

    def test_incomplete_http_error_body_is_reported(self) -> None:
        error = ds4_publication_decode_client.urllib.error.HTTPError(
            "http://127.0.0.1:1/v1/chat/completions",
            503,
            "Service Unavailable",
            {},
            _TruncatedErrorBody(),
        )
        with mock.patch.object(
            ds4_publication_decode_client.urllib.request,
            "urlopen",
            side_effect=error,
        ):
            result = ds4_publication_decode_client.stream_request(
                "http://127.0.0.1:1", "dflash", "prompt", max_tokens=2
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], 503)
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


class QualifierPreflightTests(unittest.TestCase):
    def run_qualifier(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        fixture = str(Path(__file__).resolve())
        environment = os.environ.copy()
        environment.pop("HIP_VISIBLE_DEVICES", None)
        environment.pop("ROCR_VISIBLE_DEVICES", None)
        environment.update(
            {
                "TARGET_MODEL": fixture,
                "DRAFT_MODEL": fixture,
                "HOTNESS_CSV": fixture,
                "SERVER_BIN": sys.executable,
                "TOKENIZER_HARNESS": sys.executable,
                **overrides,
            }
        )
        return subprocess.run(
            ["bash", str(QUALIFIER)],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )

    def test_leading_zero_integer_is_rejected(self) -> None:
        result = self.run_qualifier(PORT="08")

        self.assertEqual(result.returncode, 2)
        self.assertIn("decimal integer", result.stderr)

    def test_reordered_visible_devices_are_rejected(self) -> None:
        result = self.run_qualifier(HIP_VISIBLE_DEVICES="1,0")

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be unset or exactly 0,1", result.stderr)

    def test_nonnumeric_q5_mode_is_rejected_cleanly(self) -> None:
        result = self.run_qualifier(Q5_VERIFY="invalid")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Q5_VERIFY must be 0 or 1", result.stderr)
        self.assertNotIn("unbound variable", result.stderr)


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

    def test_compact_report_preserves_repeated_contexts(self) -> None:
        report = ds4_context_sweep.compact_group_report(
            [
                {"target_context": 2048, "measured_summary": {"n_ok": 3}},
                {"target_context": 4096, "measured_summary": {"n_ok": 3}},
                {"target_context": 2048, "measured_summary": {"n_ok": 3}},
            ]
        )

        self.assertEqual(
            [row["target_context"] for row in report],
            [2048, 4096, 2048],
        )


class RocprofTimelineTests(unittest.TestCase):
    def test_nonfinite_float_options_are_rejected(self) -> None:
        for option, value in (
            ("--bin-ms", "nan"),
            ("--window-start-s", "inf"),
            ("--window-end-s", "-inf"),
            ("--timeline-merge-gap-us", "nan"),
        ):
            with self.subTest(option=option):
                argv = ["rocprof-overlap", "unused.csv", option, value]
                with (
                    mock.patch.object(sys, "argv", argv),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    analyze_rocprof_overlap.main()

                self.assertEqual(error.exception.code, 2)

    def test_malformed_dispatch_rows_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text(
                "Agent_Id,Start_Timestamp,End_Timestamp,Kernel_Name\n"
                "gpu0,100,200,kernel0\n"
                "gpu0,invalid,250,broken\n"
                "gpu1,120,180,kernel1\n",
                encoding="utf-8",
            )
            argv = ["rocprof-overlap", str(trace), "--top", "0"]
            with (
                mock.patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                result = analyze_rocprof_overlap.main()

        self.assertEqual(result, 0)
        self.assertIsNone(analyze_rocprof_overlap.parse_dispatch({}))

    def test_window_is_clipped_before_merge_and_cap_is_per_agent(self) -> None:
        bursts = analyze_rocprof_overlap.build_timeline_bursts(
            {
                "gpu0": [(0, 95), (105, 110), (150, 160)],
                "gpu1": [(101, 104), (140, 150)],
            },
            ["gpu0", "gpu1"],
            window_start=100,
            window_end=200,
            merge_gap_ns=20,
            per_agent_limit=1,
        )

        self.assertEqual(
            bursts,
            [(101, 104, "gpu1"), (105, 110, "gpu0")],
        )


if __name__ == "__main__":
    unittest.main()
